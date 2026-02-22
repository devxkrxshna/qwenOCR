from pathlib import Path
from typing import Union
from PIL import Image
import base64
from io import BytesIO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen2VLChatHandler


class QwenOCR:
    def __init__(self, model_path: Union[str, Path], n_ctx: int = 2048, n_gpu_layers: int = -1):
        self.model_path = Path(model_path)
        chat_handler = Qwen2VLChatHandler(clip_model_path=str(self.model_path))
        self.llm = Llama(
            model_path=str(self.model_path),
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
        )

    def _image_to_base64(self, image: Union[str, Path, Image.Image]) -> str:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def extract_text(self, image: Union[str, Path, Image.Image], prompt: str = "Extract all text from this image.") -> str:
        image_b64 = self._image_to_base64(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        response = self.llm.create_chat_completion(messages=messages)
        return response["choices"][0]["message"]["content"]
