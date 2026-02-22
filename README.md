# Qwen OCR Library

Standalone OCR library using Qwen2-VL-2B without API keys.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Download the model from [HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GGUF):
   - Recommended: `qwen2-vl-2b-instruct-q4_k_m.gguf` (~1.5GB)

## Usage

```python
from qwen_ocr import QwenOCR

ocr = QwenOCR(model_path="qwen2-vl-2b-instruct-q4_k_m.gguf")
text = ocr.extract_text("image.jpg")
print(text)
```

## Parameters

- `model_path`: Path to GGUF model file
- `n_gpu_layers`: GPU layers (-1 for all, 0 for CPU only)
- `n_ctx`: Context window size (default: 2048)
