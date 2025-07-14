import os
from typing import Any
from openai import OpenAI

from agents_manager.models import OpenAi


class DeepSeek(OpenAi):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the DeepSeek model with a name and optional keyword arguments.

        Args:
            name (str): The name of the DeepSeek model (e.g., "deepseek-chat").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if current_api_key := kwargs.get("api_key") or os.environ.get(
            "DEEPSEEK_API_KEY"
        ):
            self.client = OpenAI(
                api_key=current_api_key, base_url="https://api.deepseek.com"
            )
        else:
            raise RuntimeError("Could not find Deepseek api key")
