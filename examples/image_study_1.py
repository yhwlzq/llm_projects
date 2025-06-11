from unstructured.partition.pdf import partition_pdf
from langchain_ollama import OllamaLLM
from PIL import Image
import io
import base64


def image_to_base64(image_path):
    with Image.open(image_path) as image:
        img_format = image.format if image.format else 'JPEG'
        buffered = io.BytesIO()
        image.save(buffered, format=img_format)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')
    
    
def identity_image():
    llm = OllamaLLM(model="llava:latest")
    image_str = image_to_base64("/home/mystic/PycharmProjects/llm/llm_projects/llm_evaluation/figure-15-6.jpg")
    messages = [
        {
            "role": "user",  # 必须包含 'role'（通常是 'user' 或 'assistant'）
            "content": [     # 必须包含 'content'，内容可以是列表或字符串
                {"type": "text", "text": "Please give a summary of the image provided. Be descriptive"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_str}"
                    }
                }
            ]
        }
    ]
    
    result = llm.invoke(messages)
    print(result)


if __name__ == "__main__":
    identity_image()