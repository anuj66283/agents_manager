import os
from typing import Any
from openai import OpenAI

from agents_manager.models import OpenAi


class Genai(OpenAi):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Genai model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Genai model (e.g., "gemini-2.0-flash").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid  Genai model name is required")

        if current_api_key := kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY"):
            self.client = OpenAI(
                api_key=current_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            raise RuntimeError("Could not find Genai api key")
