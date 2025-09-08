FROM python:3.10-slim

WORKDIR /app

# Install base requirements
COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

# Install metric specific requirements
COPY requirements/bertscore.txt requirements/bertscore.txt
RUN pip install --no-cache-dir -r requirements/bertscore.txt

COPY requirements/summac.txt requirements/summac.txt
RUN pip install --no-cache-dir -r requirements/summac.txt

# Copy application
COPY app app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
