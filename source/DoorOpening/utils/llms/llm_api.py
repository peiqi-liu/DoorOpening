import os
from typing import Any, Dict, Optional, Union
from PIL import Image
from typing import List
from google import genai
from google.genai import types
import textwrap
import io
import torch
import torchvision.transforms.functional as F  


class GeminiClient():
    """Simple client for creating agents using an OpenAI API call.

    Parameters:
        use_specific_objects(bool): override list of objects and have the AI only return those."""

    def __init__(
        self,
        prompt: str,
        model: str = "gemini-3-pro-preview",
    ):
        self.system_prompt = prompt
        self.model = model

        if "GOOGLE_API_KEY" in os.environ:
            self._gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            raise Exception("Gemini token has not been set up yet!")

    def __call__(
        self, command: Union[str, list], model: Optional[str] = None, verbose: bool = False
    ):
        if verbose:
            print(f"{self.system_prompt=}")
        if model is None:
            model = self.model
        if isinstance(command, str):
            command = [command]
        command = [self.system_prompt] + command
        print("command:", command)
        response = self._gemini.models.generate_content(model=self.model, contents=command)

        plan = response.text
        if verbose:
            print(f"plan={plan}")
        return plan

class GeminiAgent():
    def __init__(self, prompt: str, model: str = "gemini-3-flash-preview"):
        self.prompt = prompt
        self.model = model
        self.client = GeminiClient(prompt, model)
        self.history = []

    def query(self, image, current_state: str):
        commands: List[Any] = []
        commands.append("History commands: ")
        for item in self.history:
            commands.append(item)
        commands.append("Image Observation: ")

        im = F.to_pil_image(image.permute(2, 0, 1))
        im.show()
        im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
        commands.append(im)

        # commands.append("Current environment state: ")
        # commands.append(current_state)
        response = self.client(commands, model=self.model)
        response = textwrap.dedent(response)
        self.history.append(response)
        
        return response