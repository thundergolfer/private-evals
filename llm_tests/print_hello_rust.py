from evaluator import LLMRun, ExtractCode, RustRun, SubstringEvaluator, run_test

DESCRIPTION = (
    'Test if the model can generate a basic Rust program that prints "hello world".'
)

TAGS = ["code", "rust"]
CATEGORY = "code"

question = 'Write a Rust program that prints the string "hello world" and tell me how it works in a sentence'

answer = "hello world"

TEST = (
    question
    >> LLMRun()
    >> ExtractCode(keep_main=True)
    >> RustRun()
    >> SubstringEvaluator(answer)
)


if __name__ == "__main__":
    from llm import default_llm, default_eval_llm, default_vision_eval_llm

    print(
        run_test(
            "print-hello-rust",
            TEST,
            default_llm,
            default_eval_llm,
            default_vision_eval_llm,
        )
    )
