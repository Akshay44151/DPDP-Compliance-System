"""Compare structured-output modes on the hard policy.

Makes up to 8 real Groq calls. Run from the project root:
    python -m scripts.compare_modes
"""

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from agents.policy_agent.agent import HUMAN_PROMPT, SYSTEM_PROMPT
from agents.policy_agent.schemas import PolicyExtractionResult
from llm.config import get_llm_config
from llm.langchain_model import get_langchain_model

POLICY = Path("tests/fixtures/hard_privacy_policy.txt")
RUNS = 4
METHODS = ["function_calling", "json_schema"]


def main() -> None:
    print("model:", get_llm_config().model, "\n")
    text = POLICY.read_text(encoding="utf-8")
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    model = get_langchain_model()

    for method in METHODS:
        try:
            structured = model.with_structured_output(
                PolicyExtractionResult, method=method
            )
        except Exception as exc:
            print(f"{method}: could not build ({type(exc).__name__}) {str(exc)[:150]}\n")
            continue

        chain = prompt | structured
        ok = 0
        for i in range(1, RUNS + 1):
            try:
                raw = chain.invoke({"source": POLICY.name, "document_text": text})
                result = PolicyExtractionResult.model_validate(raw)
                ok += 1
                print(f"  {method} run {i}: OK, {len(result.claims)} claims")
            except Exception as exc:
                print(f"  {method} run {i}: FAILED ({type(exc).__name__}) {str(exc)[:120]}")
        print(f"{method}: {ok}/{RUNS} succeeded\n")


if __name__ == "__main__":
    main()