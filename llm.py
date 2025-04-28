import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MultiLLMClient:
    """A client that can interact with multiple LLM APIs (Anthropic Claude, OpenAI, Deepseek, Google Gemini)."""

    def __init__(self, provider="openai", model="gpt-4o"):
        self.provider = provider
        self.model = model
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self._google_api_key = os.getenv("GOOGLE_AISTUDIO_API_KEY")
        self._anthropic_endpoint = "https://api.anthropic.com/v1/messages"
        self._openai_endpoint = "https://api.openai.com/v1/chat/completions"
        self._deepseek_endpoint = "https://api.deepseek.com"
        self._google_endpoint = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"

    def __call__(self, prompt, json: bool = False):
        if json:
            print("WARN: ignoring json=True for now")
        if "claude" == self.provider:
            return self._call_claude(prompt)
        elif "deepseek" == self.provider:
            return self._call_deepseek(prompt)
        elif "gemini" == self.provider:
            return self._call_gemini(prompt)
        elif "openai" == self.provider:
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unsupported model: {self.provider}")

    def _call_claude(self, prompt, max_tokens=1000):
        if not self._anthropic_api_key:
            return "Error: Anthropic API key not found"

        # TODO: try 'claude-3-7-sonnet-20250219'
        import anthropic

        try:
            client = anthropic.Anthropic(api_key=self._anthropic_api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            )
            return message.content[0].text
        except Exception as e:
            return f"Error calling Claude API: {str(e)}"

    def _call_openai(self, prompt: str, max_tokens=1000):
        from openai import OpenAI

        if not self._openai_api_key:
            return "Error: OpenAI API key not found"

        try:
            client = OpenAI(api_key=self._openai_api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    def _call_deepseek(self, prompt, max_tokens=1000):
        from openai import OpenAI

        if not self._deepseek_api_key:
            raise RuntimeError("Error: Deepseek API key not found")

        try:
            client = OpenAI(
                api_key=self._deepseek_api_key, base_url=self._deepseek_endpoint
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Deepseek API: {str(e)}"

    def _call_gemini(self, prompt, max_tokens=1000):
        # TODO: try 'gemini-2.5-pro-preview-03-25'
        if not self._google_api_key:
            raise RuntimeError("Error: Google API key not found")

        try:
            from google import genai

            client = genai.Client(api_key=self._google_api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"


default_llm = MultiLLMClient(provider="openai", model="gpt-4o")
default_eval_llm = MultiLLMClient(provider="gemini", model="gemini-2.0-flash")
default_vision_eval_llm = MultiLLMClient(provider="gemini", model="gemini-2.0-flash")
