"""
AWS Lambda Function — Amazon Hugging Face Data Extractor
Fetches Amazon's public models and datasets from Hugging Face API
and saves the results as a JSON file in S3.

Environment Variables (set in Lambda configuration):
    S3_BUCKET_NAME  — name of your S3 bucket (required)
    S3_FILE_KEY     — S3 object key/path (optional, default: amazon_hf_data.json)

Dependencies: boto3 (built into Lambda), urllib (built into Python)
No external packages or Lambda layers needed.
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone

import boto3

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
    """Extract relevant performance metrics from a model entry."""
    return {
        "id": m.get("id"),
        "author": m.get("author"),
        "task": m.get("pipeline_tag"),
        "downloads": m.get("downloads", 0) or 0,
        "likes": m.get("likes", 0) or 0,
        "tags": m.get("tags", []),
        "last_modified": m.get("lastModified"),
        "created_at": m.get("createdAt"),
        "private": m.get("private", False),
    }


def build_dataset_record(d: dict) -> dict:
    """Extract relevant performance metrics from a dataset entry."""
    return {
        "id": d.get("id"),
        "author": d.get("author"),
        "downloads": d.get("downloads", 0) or 0,
        "likes": d.get("likes", 0) or 0,
        "tags": d.get("tags", []),
        "last_modified": d.get("lastModified"),
        "created_at": d.get("createdAt"),
        "private": d.get("private", False),
    }


def build_summary(models: list, datasets: list) -> dict:
    """Build a high-level summary of Amazon's HF presence."""
    total_model_downloads = sum(m["downloads"] for m in models)
    total_dataset_downloads = sum(d["downloads"] for d in datasets)
    total_model_likes = sum(m["likes"] for m in models)
    total_dataset_likes = sum(d["likes"] for d in datasets)

    # Count models per task
    tasks: dict = {}
    for m in models:
        task = m.get("task") or "other"
        tasks[task] = tasks.get(task, 0) + 1

    top_model = max(models, key=lambda m: m["downloads"], default=None)
    top_dataset = max(datasets, key=lambda d: d["downloads"], default=None)

    return {
        "total_models": len(models),
        "total_datasets": len(datasets),
        "total_model_downloads": total_model_downloads,
        "total_dataset_downloads": total_dataset_downloads,
        "total_model_likes": total_model_likes,
        "total_dataset_likes": total_dataset_likes,
        "models_by_task": tasks,
        "top_downloaded_model": top_model["id"] if top_model else None,
        "top_downloaded_model_count": top_model["downloads"] if top_model else 0,
        "top_downloaded_dataset": top_dataset["id"] if top_dataset else None,
        "top_downloaded_dataset_count": top_dataset["downloads"] if top_dataset else 0,
    }


def save_to_s3(data: dict, bucket: str, key: str) -> str:
    """Save data as a JSON file to S3 and return the S3 URI."""
    s3 = boto3.client("s3")
    body = json.dumps(data, indent=2, default=str).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def lambda_handler(event, context):
    """Lambda entry point."""
    bucket = os.environ.get("S3_BUCKET_NAME")
    if not bucket:
        raise ValueError("Environment variable S3_BUCKET_NAME is not set.")

    # Allow the S3 key to be overridden via event payload or env var
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_key = f"huggingface/amazon/amazon_hf_data_{timestamp}.json"
    s3_key = event.get("s3_key") or os.environ.get("S3_FILE_KEY", default_key)

    print(f"Fetching Amazon models from Hugging Face...")
    raw_models = fetch_amazon_models(limit=100)
    models = [build_model_record(m) for m in raw_models]
    print(f"  Found {len(models)} models.")

    print(f"Fetching Amazon datasets from Hugging Face...")
    raw_datasets = fetch_amazon_datasets(limit=100)
    datasets = [build_dataset_record(d) for d in raw_datasets]
    print(f"  Found {len(datasets)} datasets.")

    output = {
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "author": AUTHOR,
        "summary": build_summary(models, datasets),
        "models": models,
        "datasets": datasets,
    }

    print(f"Saving results to S3: s3://{bucket}/{s3_key}")
    s3_uri = save_to_s3(output, bucket, s3_key)
    print(f"Saved successfully: {s3_uri}")

    return {
        "statusCode": 200,
        "body": {
            "message": "Data extracted and saved successfully.",
            "s3_uri": s3_uri,
            "models_count": len(models),
            "datasets_count": len(datasets),
        }
    }
