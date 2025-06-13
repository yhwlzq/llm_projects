import base64
import logging
import os.path
import time
import torch
from PIL import Image
from io import BytesIO
from typing import  List
from settings import settings
from fastapi import FastAPI, HTTPException, UploadFile, File
from transformers import CLIPModel

from  pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="CLIP Image Embedding Service")

class ImageRequest(BaseModel):
    image_64:str

class ImageEmbeddingResponse(BaseModel):
    embedding:List[float]
    model:str
    elapsed_ms:float

class CLIPImageEngine(object):

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL, trust_remote_code=True)

    async def embed_image(self,image_b64:str)->List[float]:
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            inputs = self.processor(
                image=image,
                return_tensor ='pt',
                padding=True
            ).to(self.device)
            with torch.no_grad():
                features =  self.model.get_image_features(**inputs)
                """
                maybe use the tranformer.pre_train to do directly, it is also ok
                """
                return features.cpu().numpy()[0].tolist()
        except Exception as e:
            logger.error(f"CLIP embedding failed: {e}")
            return None


encoder = CLIPImageEngine()


@app.post("/v1/embed_image", response_model=ImageEmbeddingResponse)
async def embed_image(request: ImageRequest):
    try:
        start_time = time.time()
        embedding = await encoder.embed_image(request.image_64)
        return {
            "embedding":embedding,
            "model":settings.CLIP_MODEL,
            "elapse_ms": (time.time()-start_time)*1000
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/embed_image_file")
async def embed_image_file(file:UploadFile=File(...)):
    try:
        contents = await file.read()
        image_b64 = base64.b64encode(contents).decode('utf-8')
        return await embed_image(ImageRequest(image_64=image_b64))
    except Exception as e:
        raise  HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5100)



''''
curl -X POST "http://localhost:5100/v1/embed_image" \
     -H "Content-Type: application/json" \
     -d '{"image_64":"$(base64 -w 0 example.jpg)"}'

# 通过文件上传调用
curl -X POST "http://localhost:5100/v1/embed_image_file" \
     -F "file=@example.jpg"
'''

import hashlib
import logging
from pathlib import Path
from typing import Generator, Dict, List, Union
from PIL import Image
from transformers import CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from processor.settings import settings


class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = CLIPModel.from_pretrained(
            settings.CLIP_MODEL,
            trust_remote_code=True
        )
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_image_caption(self, image_path: str) -> str:
        """使用BLIP生成图像描述"""
        raw_image = Image.open(image_path).convert('RGB')

        # 无条件图像描述
        inputs = self.processor(raw_image, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption

    def process(self, image_paths: Union[Path, List[Path]], batch_size:int = 32) -> Generator[Dict[str, any], None, None]:
        if isinstance(image_paths, (str, Path)):
            image_paths = [Path(image_paths)]
        else:
            image_paths = [Path(p) for p in image_paths]

        # Process images in batches for efficiency
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            try:
                images = []
                valid_paths = []
                for img_path in batch:
                    if not os.path.exists(img_path):
                        continue
                    try:
                        img = Image.open(img_path)
                        images.append(img)
                        valid_paths.append(img_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to load image {img_path}: {str(e)}")
                        continue

                if not images:
                    continue

                # Generate embeddings for the batch
                embeddings = self.model.encode(images)

                # Yield each image's data with metadata
                for img_path, embedding in zip(valid_paths, embeddings):
                    file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()

                    yield {
                        'text': "",  #self.generate_image_caption(img_path)
                        'embedding': embedding.tolist(),  # Convert to list for JSON serialization
                        'metadata': {
                            'source': str(img_path),
                            'file_name': img_path.name,
                            'doc_hash': file_hash,
                            'file_size': img_path.stat().st_size,
                            'file_type': img_path.suffix.lower(),
                            'dimensions': f"{images[0].width}x{images[0].height}" if images else "unknown"
                        }
                    }

            except Exception as e:
                self.logger.error(f"Failed to process image batch starting at {i}: {str(e)}")
                # Continue with next batch even if one fails
                continue