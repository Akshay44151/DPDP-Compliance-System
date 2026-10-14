
from llm.groq_client import GroqLLMClient


def main():
    client = GroqLLMClient()

    response = client.generate(
        system_prompt="You are a helpful assistant. Keep answers short.",
        user_prompt="Reply with exactly: Groq connection successful.",
    )

    print("Groq API response:")
    print(response)


if __name__ == "__main__":
    main()
