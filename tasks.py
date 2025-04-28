import json
import os
import glob
import pathlib
from dotenv import load_dotenv
from invoke import task

from llm import MultiLLMClient
from evaluator import TestGroup, create_html_report, run_test

load_dotenv()
PROVIDER_TO_DEFAULT_MODEL = {
    "claude": "claude-3-5-sonnet-20240620",
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
}


def call_llm(prompt, provider) -> None:
    client = MultiLLMClient()

    if provider == "claude" or provider == "all":
        model = PROVIDER_TO_DEFAULT_MODEL["claude"]
        client = MultiLLMClient(provider="claude", model=model)
        print(f"\n===== Claude ({model}) Response =====")
        print(client(prompt))

    if provider == "openai" or provider == "all":
        model = PROVIDER_TO_DEFAULT_MODEL["openai"]
        client = MultiLLMClient(provider="openai", model=model)
        print(f"\n===== OpenAI ({model}) Response =====")
        print(client(prompt))

    if provider == "deepseek" or provider == "all":
        model = PROVIDER_TO_DEFAULT_MODEL["deepseek"]
        client = MultiLLMClient(provider="deepseek", model=model)
        print(f"\n===== Deepseek ({model}) Response =====")
        print(client(prompt))

    if provider == "gemini" or provider == "all":
        model = PROVIDER_TO_DEFAULT_MODEL["gemini"]
        client = MultiLLMClient(provider="gemini", model=model)
        print(f"\n===== Gemini ({model}) Response =====")
        print(client(prompt))


def import_test(test: str):
    full_module_name = f"llm_tests.{test}"
    module = __import__(full_module_name, fromlist=["TEST", "CATEGORY"])
    t = getattr(module, "TEST")
    category = getattr(module, "CATEGORY")
    return t, category


def run_benchmarks(providers: list[str], results_path: str, reports_path: str):
    results = []
    test_files = glob.glob(os.path.join("llm_tests", "*.py"))
    if not (
        test_modules := [os.path.splitext(os.path.basename(f))[0] for f in test_files]
    ):
        raise RuntimeError("No test files found in llm_tests directory")

    def slugify(s: str) -> str:
        return s.lower().replace("_", "-").strip()

    results = []
    # TODO(Jonathon): this should run in parallel.
    for module_name in test_modules:
        test, category = import_test(module_name)
        test_slug = slugify(module_name)
        for provider in providers:
            model = PROVIDER_TO_DEFAULT_MODEL[provider]
            print(f"Running test {test} with provider {provider} and model {model}")
            llm = MultiLLMClient(provider=provider, model=model)
            eval_llm = MultiLLMClient(provider="gemini", model="gemini-2.0-flash")
            vision_eval_llm = MultiLLMClient(
                provider="gemini", model="gemini-2.0-flash"
            )
            result = run_test(
                test_slug=test_slug,
                category=category,
                test=test,
                llm=llm,
                eval_llm=eval_llm,
                vision_eval_llm=vision_eval_llm,
            )
            results.append(result)
            print(json.dumps(result.model_dump_json(), indent=4))

    pathlib.Path(results_path).write_text(
        TestGroup(tests=results).model_dump_json(indent=4)
    )
    print("Results written to results.json")

    html = create_html_report(results)
    pathlib.Path(reports_path).write_text(html)
    print("Report written to report.html")


@task
def prompt(ctx, prompt: str, provider: str = "all"):
    """
    Run a text prompt against one or all providers and print the result.
    This is just a sanity check command to ensure that the LLM clients are working.
    """
    _ = ctx
    if provider != "all" and provider not in PROVIDER_TO_DEFAULT_MODEL:
        raise ValueError(f"Invalid provider: {provider}")
    call_llm(prompt, provider)


@task
def run(ctx, test: str = "print_hello_python"):
    """
    Run a single LLM test against one provider and print the result.
    Test arguments should be the module name of the test inside the `llm_tests` package.

    E.g. print_hello_python.py → --test="print_hello_python"
    """
    _ = ctx
    t, category = import_test(test)
    llm = MultiLLMClient(provider="gemini", model="gemini-2.0-flash")
    eval_llm = MultiLLMClient(provider="gemini", model="gemini-2.0-flash")
    vision_eval_llm = MultiLLMClient(provider="gemini", model="gemini-2.0-flash")
    result = run_test(
        test_slug=test,
        category=category,
        test=t,
        llm=llm,
        eval_llm=eval_llm,
        vision_eval_llm=vision_eval_llm,
    )
    print(result.model_dump_json(indent=4))


@task
def run_all(ctx, provider: str | None = None):
    """
    Run all LLM tests against one or all providers and generate a report.
    """
    providers = PROVIDER_TO_DEFAULT_MODEL if provider is None else [provider]
    if provider and provider not in PROVIDER_TO_DEFAULT_MODEL:
        raise ValueError(f"Invalid provider: {provider}")
    run_benchmarks(providers, "results.json", "report.html")


@task
def generate_report(ctx):
    """Take an existing saved results JSON file and generate from it an HTML report."""
    _ = ctx
    with open("results.json", "r") as f:
        results_json = json.load(f)
        test_group = TestGroup.model_validate(results_json)
    html = create_html_report(test_group.tests)
    with open("report.html", "w") as f:
        f.write(html)
