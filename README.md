# evaluator-test

FastAPI service providing evaluation metrics for text generation.

## Features
- Supports BERTScore and SummaC metrics.
- Modular metric registry to make it easy to add more metrics.
- Per-metric requirement files under `requirements/` for independent version management.
- Dockerfile for containerized deployment.

## Usage
Install base and metric requirements:
```bash
pip install -r requirements/base.txt
pip install -r requirements/bertscore.txt
pip install -r requirements/summac.txt
```
Run the API:
```bash
uvicorn app.main:app --reload
```

### Configuration

Set `METRIC_TIMEOUT` (seconds) to control how long each metric evaluation is allowed to run.
The default is 10 seconds.

## API
`POST /evaluate` expects a JSON body:
```json
{
  "candidate": "generated text",
  "reference": "reference text",
  "metrics": ["bertscore", "summac"]
}
```
Response example:
```json
{
  "bertscore": 0.89,
  "summac": 0.75
}
```

## Docker
Build and run with Docker:
```bash
docker build -t evaluator .
docker run -p 8000:8000 evaluator
```
