FROM nvidia/cuda:12.1-base
RUN pip install vllm fastapi uvicorn pillow transformers
COPY backend/ /app/
WORKDIR /app
CMD ["/bin/bash", "run.sh"]