## Copyright (C) 2024, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import io
import inspect
import re
import textwrap


from PIL import Image

import container_controller
from container_controller import invoke_container


from pydantic import BaseModel, validator
from datetime import datetime


class TestResult(BaseModel):
    provider: str
    model: str
    # Category is a high-level grouping of tests.
    # "code" is for code generation tasks.
    # "writing" is for writing tasks.
    # "recommendation" is for recommendation tasks.
    test_category: str
    test_slug: str
    created_at: datetime
    prompt: str
    response: str
    success: bool | None = None
    rating_10: int | None = None

    @validator("rating_10")
    def validate_rating(cls, v):
        if v is None:
            return None
        if not 0 <= v <= 10:
            raise ValueError("rating_10 must be between 0 and 10")
        return v

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class TestGroup(BaseModel):
    tests: list[TestResult]


## Constants that define which model we're supposed to be using:
LLM = "llm"  # The LLM under evaluation
EVAL_LLM = "eval_llm"  # A good LLM that can act as a judge
VISION_EVAL_LLM = "vision_eval_llm"  # And a good judge for vision tasks
PYTHON_ENV = "python3"  # The version of python to use


class Env:
    """
    An environment that holds the local variables for each test case.
    """

    is_setup: bool = False
    # The container we're running the tests in
    container = None
    # The container image to use with the container.
    image_ref = None


class Reason:
    """
    A class to keep track of the solution path of a test.
    """

    def __init__(self, node, children):
        self.node = node
        self.children = children

    def __repr__(self):
        return repr((self.node, self.children))


class Node:
    """
    A node forms the operations in the computation graph for evaluating a test case;
    the most important object in this file. A test case might look like

        Node1 >> Node2 >> (Node3 & Node4)

    Each of these operators that connects nodes return a new node. So this graph
    would be equivalent to writing:

        ThenNode(ThenNode(Node1, Node2), AndNode(Node3, Node4))

    Once the computation graph has been constructed, evaluation is performed by
    calling __call__ on the root node, that then passes off the evalaution process
    as defined by each of the node types.
    """

    def __init__(self, runner):
        """
        Many sub-classes take a single argument, the runner, which is a function
        that should be executed for performing this node's computation.
        """
        self.runner = runner

    def setup(self, env, conv, llm, eval_llm, vision_eval_llm):
        """
        Once the graph has been constructed, before running __call__ to evaluate
        the test case, we run setup() on each of the nodes to pass all the
        necessary context.
        """
        self.env = env
        self.conv = conv
        self.llm = llm
        self.eval_llm = eval_llm
        self.vision_eval_llm = vision_eval_llm

    def __call__(self, orig_output=""):
        """
        Evaluate the test case, starting at this node. This is the main entry
        point for the evaluation process.

        Returns two arguments:
        1. The output of the current node that should be passed to the next node.
        2. A Reason object that explains how the output was generated for debugging.

        """
        raise NotImplementedError()

    def __rshift__(self, other_node):
        """
        Add the >> operator, which creates a ThenNode.
        Wrap any strings in a StringNode first, to allow for code like

            SetupNode >> "command to run" >> LLMRunNode
        """

        if isinstance(other_node, str):
            other_node = StringNode(other_node)
        return ThenNode(self, other_node)

    def __rrshift__(self, other_node):
        """
        If a string is the first node, we need to special case the
        rrshift operator, since we can't override the string class.
        Allows the (very common) pattern of

            "command to run" >> LLMRunNode
        """
        if isinstance(other_node, str):
            other_node = StringNode(other_node)
        return ThenNode(other_node, self)

    def __and__(self, other_node):
        return AndNode(self, other_node)

    def __or__(self, other_node):
        return OrNode(self, other_node)

    def __invert__(self):
        return NotNode(self)


class StringNode(Node):
    def __init__(self, string):
        """
        A boring node, just returns the string.
        """
        self.string = string

    def __call__(self, orig_output=""):
        """
        Just pass whatever the provided constant string is to the next node.
        """
        yield self.string, Reason(type(self), self.string)


class ThenNode(Node):
    """
    Perform two operations in sequence. The output of node1 is passed to node2.
    """

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def setup(self, env, conv, llm, eval_llm, vision_eval_llm):
        super().setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.node1.setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.node2.setup(
            env=env,
            conv=conv,
            llm=llm,
            eval_llm=eval_llm,
            vision_eval_llm=vision_eval_llm,
        )

    def __call__(self, orig_output=None):
        for output1, response1 in self.node1(orig_output):
            for output2, response2 in self.node2(output1):
                yield output2, Reason(type(self), (response1, response2))


class AndNode(ThenNode):
    """
    An evaluation node that returns true if both outputs are true.
    """

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def __call__(self, orig_output):
        for output1, txt1 in self.node1(orig_output):
            for output2, txt2 in self.node2(orig_output):
                yield (
                    output1 and output2,
                    Reason(type(self), (txt1, txt2, output1 and output2)),
                )


class OrNode(ThenNode):
    """
    An evaluation node that returns true if either outputs are true.
    """

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def __call__(self, orig_output):
        for output1, txt1 in self.node1(orig_output):
            for output2, txt2 in self.node2(orig_output):
                yield (
                    output1 or output2,
                    Reason(type(self), (txt1, txt2, output1 or output2)),
                )


class NotNode(Node):
    """
    An evaluation node that negates the prior answer.
    """

    def __init__(self, node1):
        self.node1 = node1

    def setup(self, env, conv, llm, eval_llm, vision_eval_llm):
        super().setup(env, conv, llm, eval_llm, vision_eval_llm)
        self.node1.setup(env, conv, llm, eval_llm, vision_eval_llm)

    def __call__(self, orig_output):
        for output1, txt1 in self.node1(orig_output):
            yield not output1, Reason(type(self), [txt1, not output1])


class PyFunc(Node):
    """
    A node that just runs a python function on the prior result.
    If the code crashes then just return an error.
    """

    def __call__(self, x):
        try:
            out = self.runner(x)
            if isinstance(out, tuple):
                ok, log = out
                return [(ok, Reason(type(self), (log, ok)))]
            else:
                return [(out, Reason(type(self), ("", out)))]
        except Exception:
            return [("", Reason(type(self), ["Error", False]))]


class Echo(Node):
    """
    A no-op node that helps debug test cases by printing whatever's being
    passed along the pipe. Kind of like the Unix tee command.
    """

    def __init__(self):
        pass

    def __call__(self, x):
        print("ECHOING:", x)
        yield x, Reason(type(self), None)


class Setup(Node):
    """
    A node that starts up a new container environment with a specific setup file.

    Even though the argument is a method, this function needs to be able to
    extract the string representation of that function so it can be executed
    in the context of the container environment.
    """

    def __call__(self, x):
        container_controller.setup_container(self.env)
        code = inspect.getsource(self.runner)
        to_invoke = self.runner.__name__

        code = code + f"\n\n{to_invoke}()"
        out = invoke_container(
            self.env, {"setup.py": code.encode()}, [PYTHON_ENV, "setup.py"]
        )

        return [(out, Reason(type(self), None))]


class PyEvaluator(Node):
    """
    A node that runs a Python program within the container environment to judge whether
    or not the test case is solved.

    Even though the argument is a method, this function needs to be able to
    extract the string representation of that function so it can be executed
    in the context of the container environment.
    """

    def __call__(self, x):
        code = inspect.getsource(self.runner)
        to_invoke = self.runner.__name__

        code = code + f"\n\nprint('final: ' + str({to_invoke}()))"
        out = invoke_container(
            self.env, {"check.py": code.encode()}, [PYTHON_ENV, "check.py"]
        )

        return [("final: True" in out, Reason(type(self), [out, "final: True" in out]))]


class SubstringEvaluator(Node):
    """
    An evaluation node that checks if a substring is in the output.
    """

    def __init__(self, substr, lower=False):
        self.substr = substr
        self.lower = lower

    def __call__(self, output):
        if self.lower:
            cond = self.substr.lower() in output.lower()
        else:
            cond = self.substr in output

        if cond:
            yield True, Reason(type(self), [self.substr, True])
        else:
            yield False, Reason(type(self), [self.substr, False])


class ContainsIntEvaluator(Node):
    """
    An evaluation node that checks if a given integer is in the output.
    """

    def __init__(self, num):
        self.num = num

    def __call__(self, output):
        all_integers = re.findall(r"-?[\d,]*\d+\.?\d*", output)
        all_integers = [x.replace(",", "") for x in all_integers]
        if str(self.num) in all_integers:
            yield True, Reason(type(self), [self.num, True])
        else:
            yield False, Reason(type(self), [self.num, False])


class EqualEvaluator(Node):
    """
    An evaluation node that checks if the output is equal to a given string.
    """

    def __init__(self, goal):
        self.goal = goal

    def __call__(self, output):
        if self.goal == output:
            yield True, Reason(type(self), [self.goal, True])
        else:
            yield False, Reason(type(self), [self.goal, False])


class ExtractCode(Node):
    """
    A node that extracts code from the response

    Usually you can just extract the code out of the response,
    but if the response contains multiple possible code objects,
    then this node queries the model again asking it for just the code.
    """

    def __init__(self, keep_main=False, postfix="", manual=None, lang=None):
        self.keep_main = keep_main
        self.postfix = postfix
        self.manual = manual
        self.lang = lang

    def try_extract(self, output):
        output = re.sub("```[a-z]*", "```", output)
        if "```" in output:
            ans = output.split("```")[1] + "\n" + self.postfix
        else:
            ans = output + "\n" + self.postfix
        yield ans

    def __call__(self, orig_output):
        if orig_output.count("```") == 2:
            for maybe in self.try_extract(orig_output):
                yield maybe, Reason(type(self), maybe)
            return

        language = ""
        if self.lang is not None:
            language = f"(in {self.lang})"

        if self.manual is not None:
            output = self.llm(self.manual.replace("<A>", orig_output))
        elif self.keep_main:
            assert self.postfix == ""
            output = self.llm(
                f"Take the below answer to my programming question {language} and return just the complete code in a single file so I can copy and paste it into an editor and directly run it. Include any header and main necessary so I can run it by copying this one file. DO NOT MODIFY THE CODE OR WRITE NEW CODE. Here is the code: \n"
                + orig_output
            )
        else:
            output = self.llm(
                f"Take the below answer to my programming question {language} and return just the complete code in a single file so I can copy and paste it into an editor and directly run it. Remove any test cases or example code after the function definition. Remove any main function. I will write those myself. Do include header imports. DO NOT MODIFY THE CODE OR WRITE NEW CODE. Here is the code: \n"
                + orig_output
                + (
                    "\nI will be running this code with the following helper functions:\n"
                    + self.postfix
                    if self.postfix
                    else ""
                )
            )

        for maybe in self.try_extract(output):
            yield maybe, Reason(type(self), maybe)


class PythonRun(Node):
    """
    A node that runs the output from the prior command as a python function.

    Optionally append a set of test cases to the code that's been provided.
    """

    def __init__(self, test_case=""):
        self.test_case = test_case

    def __call__(self, code):
        self.env.image_ref = "python:3.11-slim"
        code = code + "\n\n" + self.test_case

        out = invoke_container(
            self.env,
            {"main.py": code.encode()},
            [PYTHON_ENV, "main.py"],
        )
        yield out, Reason(type(self), (code, out))


class BashRun(Node):
    """
    A node that runs the output from the prior command as a bash script.
    """

    def __init__(self, test_case="", args=[]):
        self.test_case = test_case
        self.args = args

    def __call__(self, code):
        code = code + "\n\n" + self.test_case

        out = invoke_container(
            self.env, {"main.sh": code.encode()}, ["bash", "main.sh", *self.args]
        )
        yield out, Reason(type(self), (code, out))


class TerminalRun(Node):
    """
    A node that directly runs a command line argument in the terminal.
    """

    def __init__(self):
        return

    def __call__(self, code):
        if code:
            out = invoke_container(
                self.env, {"main.sh": code.encode()}, ["bash", "main.sh"]
            )
        else:
            out = ""
        yield out, Reason(type(self), (code, out))


class RustRun(Node):
    """
    A node that compiles and runs the output Rust code from the prior command.

    Optionally append a set of test cases to the code that's been provided.
    """

    def __init__(self, test_case=""):
        self.test_case = test_case

    def __call__(self, code):
        if "fn main" in code and "fn main" in self.test_case:
            code = code.replace("fn main", "fn __delete_this__main")

        code = code + "\n\n" + self.test_case

        self.env.image_ref = "rust:slim-bullseye"
        out = invoke_container(
            self.env,
            {
                "main.rs": code.encode(),
                "main.sh": "echo 'compiling'\nrustc -o a.out main.rs\n./a.out\necho 'done'".encode(),
            },
            ["bash", "main.sh"],
        )
        yield out, Reason(type(self), (code, out))


class CRun(Node):
    """
    A node that runs the output from the prior command as a c function.

    Optionally append a set of test cases to the code that's been provided.
    """

    def __init__(self, test_case="", gccflags="", argv=""):
        self.test_case = test_case
        self.gccflags = gccflags
        self.argv = argv

    def __call__(self, code):
        if "int main" in code and "int main" in self.test_case:
            code = code.replace("int main", "int __delete_this__main")

        code = code + "\n\n" + self.test_case

        out = invoke_container(
            self.env,
            {
                "main.c": code.encode(),
                "main.sh": f"gcc -o a.out main.c -lm {self.gccflags}\n./a.out {self.argv}".encode(),
            },
            ["bash", "main.sh"],
        )
        yield out, Reason(type(self), (code, out))


class LLMRun(Node):
    """
    A node to invoke a language model on any given text.

    This is the core function that allows us to evaluate the capabilities of any model.
    """

    def __init__(self, check_prompt="<A>", llm=LLM, json=False):
        self.check_prompt = check_prompt
        self.which_llm = llm
        self.json = json

    def __call__(self, output):
        llm = getattr(self, self.which_llm)
        to_send = self.check_prompt.replace("<A>", output)
        out = llm(to_send, json=self.json)

        yield out, Reason(type(self), (to_send, out))


class LLMConversation(Node):
    """
    A node to invoke a language model on any given text, but keeps state.

    This node allows us to send messages that refer to prior messages, whereas
    LLMRun is just a stateless operation.
    """

    def __init__(self, check_prompt="<A>"):
        self.check_prompt = check_prompt

    def __call__(self, output):
        to_send = self.check_prompt.replace("<A>", output)
        out = self.conv(to_send)
        yield out, Reason(type(self), (to_send, out))


class LLMVisionRun(Node):
    """
    A node to evalaute an image output from a prior operation. Invokes the
    vision evaluation model.
    """

    def __init__(self, check_prompt="<A>", llm=VISION_EVAL_LLM):
        self.check_prompt = check_prompt
        self.which_llm = llm

    def __call__(self, output):
        llm = getattr(self, self.which_llm)
        try:
            if isinstance(output, bytes):
                img = Image.open(io.BytesIO(output))
            else:
                img = output
            out = llm(self.check_prompt, add_image=img, max_tokens=512)
        except Exception as e:
            out = str(e)
        yield out, Reason(type(self), (self.check_prompt, out))


class Conversation:
    """
    An object that keeps track of the conversation history between the
    model and the test case prior questions/steps.
    """

    def __init__(self, llm, preample=""):
        self.llm = llm
        self.history = []
        self.preample = preample

    def __call__(self, msg):
        if len(self.history) == 0:
            msg = self.preample + msg
        self.history.append(msg)
        output = self.llm(self.history)
        self.history.append(output)
        return output

    def __repr__(self):
        return "Conversation(" + repr(self.history) + ")"


def run_test(
    test_slug: str,
    category: str,
    test,
    llm=None,
    eval_llm=None,
    vision_eval_llm=None,
    print_results: bool = True,
) -> TestResult:
    """
    A helper function to run just one specific test case.
    Used to debug tests by running each file directly.
    """
    env = Env()
    test.setup(env, Conversation(llm), llm, eval_llm, vision_eval_llm)

    ok = False
    for success, output in test():
        if success:
            ok = True
            break

    fmt = format_markdown(output)
    while "\n\n" in fmt:
        fmt = fmt.replace("\n\n", "\n")
    fmt = fmt.replace("\n#", "\n\n#")
    if print_results:
        print(fmt)

    if env.container:
        container_controller.async_kill_container(env.container)

    return TestResult(
        provider=llm.provider,
        model=llm.model,
        test_category=category,
        test_slug=test_slug,
        created_at=datetime.now(),
        prompt=fmt,
        response=fmt,
        success=ok,
        rating_10=None,  # Filled out manually
    )


def make_python_test(q_and_a, header=""):
    qs = [header]

    for q, a in q_and_a:
        qs.append(f"""
answer = {q}
expected = {a}
assert answer == expected, f'Wrong answer; got {{answer}} instead of {{expected}}'""")
    qs.append("print('All tests passed')")

    return "\n".join(qs), "All tests passed"


def make_c_test(q_and_a, header="", extra_methods=""):
    qs = []

    qs.append(
        "#include<stdio.h>\n#include<stdlib.h>\n" + extra_methods + "\nint main() {"
    )
    qs.append(header)
    for q, a in q_and_a:
        qs.append(f"""
int answer = {q};
int expected = {a};
if (answer != expected) {{
    printf("Wrong answer; got %d instead of %d.\\n", answer, expected);
    exit(1);
}}""")
    qs.append('printf("All tests passed\\n");')

    qs.append("}")

    return "\n".join(qs), "All tests passed"


def create_html_report(results: list[TestResult]) -> str:
    # Group results by category and test_slug
    grouped_results = {}
    for result in results:
        category = result.test_category
        test_slug = result.test_slug

        if category not in grouped_results:
            grouped_results[category] = {}

        if test_slug not in grouped_results[category]:
            grouped_results[category][test_slug] = []

        grouped_results[category][test_slug].append(result)
    with open("templates/report_template.html", "r") as f:
        report = f.read()

    total_tests = 0
    # Calculate pass/total for each provider
    provider_stats = {
        "claude": {"passed": 0, "total": 0, "ratings": []},
        "deepseek": {"passed": 0, "total": 0, "ratings": []},
        "gemini": {"passed": 0, "total": 0, "ratings": []},
        "openai": {"passed": 0, "total": 0, "ratings": []},
    }

    for category_tests in grouped_results.values():
        for tests in category_tests.values():
            total_tests += 1
            for test in tests:
                provider = test.provider
                provider_stats[provider]["total"] += 1
                if test.success:
                    provider_stats[provider]["passed"] += 1
                if test.rating_10 is not None:
                    provider_stats[provider]["ratings"].append(test.rating_10)

    insert_marker = "<!-- REPLACE TESTS HERE -->"
    insert_html = "\n"
    for category, tests_map in grouped_results.items():
        insert_html += f'<div class="category-title">{category}</div>'
        for test_slug, tests in tests_map.items():
            provider_to_test = {test.provider: test for test in tests}
            check = '<div class="feature-value"><span class="check">✓</span></div>'
            cross = '<div class="feature-value"><span class="cross">✕</span></div>'
            scores = ""
            # IMPORTANT: Lexicographic order
            scores += f"<div class='feature-value'>{provider_to_test['claude'].rating_10}</div>"
            scores += check if provider_to_test["claude"].success else cross
            scores += f"<div class='feature-value'>{provider_to_test['deepseek'].rating_10}</div>"
            scores += check if provider_to_test["deepseek"].success else cross
            scores += f"<div class='feature-value'>{provider_to_test['gemini'].rating_10}</div>"
            scores += check if provider_to_test["gemini"].success else cross
            scores += f"<div class='feature-value'>{provider_to_test['openai'].rating_10}</div>"
            scores += check if provider_to_test["openai"].success else cross

            insert_html += textwrap.dedent(f"""
            <div class="feature-row">
                <div class="feature-name">{test_slug}</div>
                {scores}
            </div>
            """)
        insert_html += "\n"
    # IMPORTANT: Lexicographic order
    score_items = (
        """
        <div class="eval-header-item">
          <h4>Rating /10</h4>
        </div>
        <div class="eval-header-item">
          <h4>Pass/Fail</h4>
        </div>
    """
        * 4
    )
    header = textwrap.dedent(f"""
    <div class="eval-header">
        <div class="eval-header-item"></div>
        <div class="provider-header" style="grid-column: 2 / span 2">Claude 3.5 Sonnet</div>
        <div class="provider-header" style="grid-column: 4 / span 2">Deepseek</div>
        <div class="provider-header" style="grid-column: 6 / span 2">Gemini 2.5 Pro</div>
        <div class="provider-header" style="grid-column: 8 / span 2">OpenAI GPT 4o</div>
    </div>
                             
    <div class="eval-header">
        <div class="eval-header-item">
          <h2>Name</h2>
        </div>
        {score_items}
      </div>
    """)
    r = report.replace("<!-- PLACE HEADER HERE -->", header)
    r = r.replace(insert_marker, insert_html)
    
    # Calculate average ratings for each provider
    avg_ratings = {}
    for provider, stats in provider_stats.items():
        if stats["ratings"]:
            avg_ratings[provider] = sum(stats["ratings"]) / len(stats["ratings"])
        else:
            avg_ratings[provider] = "N/A"
    
    summary = f"""<div class="summary-row">
        <div class="feature-name" style="opacity: 0.8">Aggregates</div>
        <div class="feature-value">{avg_ratings['claude'] if avg_ratings['claude'] != "N/A" else "N/A"}</div>
        <div class="feature-value">{provider_stats['claude']['passed']}/{provider_stats['claude']['total']}</div>
        <div class="feature-value">{avg_ratings['deepseek'] if avg_ratings['deepseek'] != "N/A" else "N/A"}</div>
        <div class="feature-value">{provider_stats['deepseek']['passed']}/{provider_stats['deepseek']['total']}</div>
        <div class="feature-value">{avg_ratings['gemini'] if avg_ratings['gemini'] != "N/A" else "N/A"}</div>
        <div class="feature-value">{provider_stats['gemini']['passed']}/{provider_stats['gemini']['total']}</div>
        <div class="feature-value">{avg_ratings['openai'] if avg_ratings['openai'] != "N/A" else "N/A"}</div>
        <div class="feature-value">{provider_stats['openai']['passed']}/{provider_stats['openai']['total']}</div>
      </div>"""
    r = r.replace("<!-- PLACE SUMMARY HERE -->", summary)
    return r


def format_markdown(reason, indent=0):
    """
    Convert a Reason object into a markdown string that explains how we got to the result.
    """

    def fix(x):
        if not isinstance(x, str):
            x = str(x)
        if x.count("```") % 2 == 1:
            x = x + "\n```"
        return "\n".join("> " + line for line in x.split("\n"))

    pounds = "#" * (indent + 1)
    if reason.node in (AndNode, OrNode):
        word = "all" if reason.node == AndNode else "any"
        out = f"{pounds} Check if {word} of the following conditions are true:\n"
        same_type = reason.node

        and_node = reason
        while and_node.node == same_type:
            out += format_markdown(and_node.children[1], indent + 1) + "\n"
            and_node = and_node.children[0]
        out += format_markdown(and_node, indent + 1) + "\n"

        out += f"\n\n{pounds}# Final Answer: {reason.children[2]}"
        return out
    elif reason.node == NotNode:
        return f"{pounds} Check this condition is not true:\n{format_markdown(reason.children[0], indent+1)}\n\n{pounds}# Final Answer: {reason.children[1]}"
    elif reason.node == ThenNode:  # has to be after and/or because children
        return (
            format_markdown(reason.children[0], indent)
            + "\n"
            + format_markdown(reason.children[1], indent)
        )
    elif reason.node == StringNode:
        return f"{pounds} Initial Query\n{fix(reason.children.strip())}\n"
    elif reason.node == Setup:
        return f"{pounds} Container Setup\nI have setup the docker container to run the model evaluation."
    elif reason.node in (LLMRun, LLMVisionRun, LLMConversation):
        return f"{pounds} LLM Generation\n#{pounds} Query\n{fix(reason.children[0].strip())}\n#{pounds} Output\n{fix(reason.children[1].strip())}\n"
    elif reason.node in (
        PythonRun,
        CRun,
        RustRun,
        BashRun,
        TerminalRun,
    ):
        return f"{pounds} Run Code Interpreter\nRunning the following program:\n> ```\n{fix(reason.children[0].strip())}\n> ```\nAnd got the output:\n```\n{reason.children[1]}\n```\n"
    elif reason.node == ExtractCode:
        return f"{pounds} Extract Code\nI extracted the following code from that output:\n> ```\n{fix(reason.children.strip())}\n> ```\n"
    elif reason.node == SubstringEvaluator:
        return f"{pounds} Substring Evaluation\nTesting if the previous output contains the string `{reason.children[0]}`: {reason.children[1]}\n"
    elif reason.node == EqualEvaluator:
        return f"{pounds} Equal Evaluation\nTesting if the previous output equals the string `{reason.children[0]}`: {reason.children[1]}\n"
    elif reason.node == ContainsIntEvaluator:
        return f"{pounds} Contains Int Evaluation\nTesting if the previous output contains the integers `{reason.children[0]}`: {reason.children[1]}\n"
    elif reason.node in (PyFunc, PyEvaluator):
        return f"{pounds} PyFunc\n{fix(reason.children[0])}\nResulting in output:\n{fix(reason.children[1])}".replace(
            "\n\n", "\n"
        )
    elif reason.node == Echo:
        return ""
    else:
        return "UNKNOWN NODE TYPE: " + repr(reason.node)


# TESTS
########################################################


def test_llm_output_collection():
    test = (
        'Write a Rust program that prints the string "hello world" and tell me how it works in a sentence'
        >> LLMRun()
        >> SubstringEvaluator("hello world")
    )
    env = Env()

    FAKE_RESPONSE = "This is a fake response! Don't use it."

    def fake_llm(prompt, json: bool = False):
        return FAKE_RESPONSE

    llm = eval_llm = vision_eval_llm = fake_llm
    test.setup(env, Conversation(llm), llm, eval_llm, vision_eval_llm)
    for success, output in test():
        print(success, output)

    assert FAKE_RESPONSE in format_markdown(output)
