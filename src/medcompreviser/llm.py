from openai import OpenAI


class VLLMChatClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        api_key: str = "EMPTY",
    ):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content