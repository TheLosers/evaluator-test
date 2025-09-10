FROM python:3.10-slim

WORKDIR /app

# Install all dependencies at once to minimize layers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0" "protobuf<4.0.0" && \
    pip install --no-cache-dir "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "transformers==4.41.2" "sentencepiece==0.1.99" "kss>=5.1.0,<6.0" "bert-score==0.3.13" "fastapi>=0.68.0,<0.69.0" "uvicorn>=0.15.0,<0.16.0" "pydantic>=1.8.0,<2.0.0"

# Copy application
COPY app app

# Pre-download models to cache
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli'); AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')"

# Set environment variables for better memory management
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Run with more memory (for Docker command)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
