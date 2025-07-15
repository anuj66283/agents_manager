import os
from typing import Any
from openai import OpenAI

from agents_manager.models import OpenAi


class Anthropic(OpenAi):
    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the Anthropic model with a name and optional keyword arguments.

        Args:
            name (str): The name of the Anthropic model (e.g., "claude-3-5-sonnet-20241022").
            **kwargs (Any): Additional arguments, including optional "api_key".
        """
        super().__init__(name, **kwargs)

        if name is None:
            raise ValueError("A valid  Anthropic model name is required")

        if current_api_key := kwargs.get("api_key") or os.environ.get(
            "ANTHROPIC_API_KEY"
        ):
            self.client = OpenAI(
                api_key=current_api_key, base_url="https://api.anthropic.com/v1/"
            )
        else:
            raise RuntimeError("Could not find Anthropic api key")
