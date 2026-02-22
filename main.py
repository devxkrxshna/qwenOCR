from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qwen_ocr import QwenOCR


def main():
    # Download model from: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GGUF
    # Example: qwen2-vl-2b-instruct-q4_k_m.gguf
    
    model_path = "path/to/qwen2-vl-2b-instruct-q4_k_m.gguf"
    
    ocr = QwenOCR(model_path=model_path, n_gpu_layers=-1)
    
    image_path = "path/to/image.jpg"
    text = ocr.extract_text(image_path)
    
    print(text)


if __name__ == "__main__":
    main()
