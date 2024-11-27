from query_text import query_rag
from langchain_ollama.llms import OllamaLLM

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_good_commit_messages():
    assert query_and_validate(
        question="What should a good commit message include?",
        expected_response="""
        - A clear description of what was changed.
        - Context for why the change was made.
        - Relevant bugs or issues that were fixed.
        - Information on any additional work done.
        """,
    )


def test_code_review_rules():
    assert query_and_validate(
        question="What types of feedback can I provide during code review?",
        expected_response="""
        High level: about software design, design patterns, anti-patterns, architec-ture, suggestion of alternative implementations. 
        Low level: details like matching coding style with the surroundings of the
        file or project, indentation, naming conventions
        """,
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    # print(f"RAW RESPONSE: {response_text}")
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = OllamaLLM(model="llama3.2:1b")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()
    # print(f"Raw evaluation result: {evaluation_results_str}")
    # print(f"Cleaned evaluation result: {evaluation_results_str_cleaned}")

    print(f"PROMPT: {prompt}")

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"RESPONSE: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"RESPONSE: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
