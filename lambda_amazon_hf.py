"""
AWS Lambda Function — Amazon Hugging Face Data Extractor
Fetches Amazon's public models and datasets from Hugging Face API
and returns the results directly in the Lambda response.

No environment variables, no S3, no extra packages needed.
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone

BASE_URL = "https://huggingface.co/api"
AUTHOR = "amazon"


def fetch_from_hf(endpoint: str, params: dict) -> list:
    """Make a GET request to the Hugging Face API using urllib (no extra deps)."""
    query_string = urllib.parse.urlencode(params)
    url = f"{BASE_URL}/{endpoint}?{query_string}"
    req = urllib.request.Request(url, headers={"User-Agent": "lambda-hf-extractor/1.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_amazon_models(limit: int = 100) -> list:
    """Fetch Amazon's public models sorted by downloads."""
    return fetch_from_hf("models", {
        "author": AUTHOR,
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
    })


def fetch_amazon_datasets(limit: int = 100) -> list:
    """Fetch Amazon's public datasets sorted by downloads."""
    return fetch_from_hf("datasets", {
        "author": AUTHOR,
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
    })


def build_model_record(m: dict) -> dict:
    return {
        "id": m.get("id"),
        "task": m.get("pipeline_tag"),
        "downloads": m.get("downloads", 0) or 0,
        "likes": m.get("likes", 0) or 0,
        "last_modified": m.get("lastModified"),
        "created_at": m.get("createdAt"),
    }


def build_dataset_record(d: dict) -> dict:
    return {
        "id": d.get("id"),
        "downloads": d.get("downloads", 0) or 0,
        "likes": d.get("likes", 0) or 0,
        "last_modified": d.get("lastModified"),
        "created_at": d.get("createdAt"),
    }


def build_summary(models: list, datasets: list) -> dict:
    tasks: dict = {}
    for m in models:
        task = m.get("task") or "other"
        tasks[task] = tasks.get(task, 0) + 1

    top_model = max(models, key=lambda m: m["downloads"], default=None)
    top_dataset = max(datasets, key=lambda d: d["downloads"], default=None)

    return {
        "total_models": len(models),
        "total_datasets": len(datasets),
        "total_model_downloads": sum(m["downloads"] for m in models),
        "total_dataset_downloads": sum(d["downloads"] for d in datasets),
        "total_model_likes": sum(m["likes"] for m in models),
        "total_dataset_likes": sum(d["likes"] for d in datasets),
        "models_by_task": tasks,
        "top_downloaded_model": top_model["id"] if top_model else None,
        "top_downloaded_model_downloads": top_model["downloads"] if top_model else 0,
        "top_downloaded_dataset": top_dataset["id"] if top_dataset else None,
        "top_downloaded_dataset_downloads": top_dataset["downloads"] if top_dataset else 0,
    }


def lambda_handler(event, context):
    """Lambda entry point — returns all results directly in the response."""
    print("Fetching Amazon models from Hugging Face...")
    models = [build_model_record(m) for m in fetch_amazon_models(limit=100)]
    print(f"  Found {len(models)} models.")

    print("Fetching Amazon datasets from Hugging Face...")
    datasets = [build_dataset_record(d) for d in fetch_amazon_datasets(limit=100)]
    print(f"  Found {len(datasets)} datasets.")

    return {
        "statusCode": 200,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "author": AUTHOR,
        "summary": build_summary(models, datasets),
        "models": models,
        "datasets": datasets,
    }
