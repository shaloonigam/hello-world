"""
AWS Lambda Function — Amazon Hugging Face Data Extractor
Fetches ALL available fields for Amazon's public models and datasets
from the Hugging Face API and saves the result as JSON to S3.

S3 destination: s3://devex-jarvis-export-data-prod/hugging_face/
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone

import boto3

BASE_URL = "https://" + "huggingface.co/api"
AUTHOR = "amazon"
S3_BUCKET = "devex-jarvis-export-data-prod"
S3_FOLDER = "hugging_face"

# Extra fields that must be explicitly requested via the expand parameter
def fetch_from_hf(endpoint: str, params: dict) -> list:
    """Make a GET request to the Hugging Face API."""
    query_string = urllib.parse.urlencode(params)
    url = BASE_URL + "/" + endpoint + "?" + query_string
    req = urllib.request.Request(url, headers={"User-Agent": "lambda-hf-extractor/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_amazon_models(limit: int = 100) -> list:
    """Fetch Amazon's public models."""
    return fetch_from_hf("models", {
        "author": AUTHOR,
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
    })


def fetch_amazon_datasets(limit: int = 100) -> list:
    """Fetch Amazon's public datasets."""
    return fetch_from_hf("datasets", {
        "author": AUTHOR,
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
    })


def build_model_record(m: dict) -> dict:
    """Extract all available fields from a model entry."""
    safetensors = m.get("safetensors") or {}
    transformers_info = m.get("transformersInfo") or {}

    return {
        # Identity
        "id": m.get("id"),
        "author": m.get("author"),
        "sha": m.get("sha"),

        # Task & Framework
        "pipeline_tag": m.get("pipeline_tag"),
        "library_name": m.get("library_name"),
        "tags": m.get("tags", []),
        "base_models": m.get("base_models", []),

        # Engagement Metrics
        "downloads": m.get("downloads", 0) or 0,
        "downloads_all_time": m.get("downloads_all_time", 0) or 0,
        "likes": m.get("likes", 0) or 0,
        "trending_score": m.get("trending_score"),
        "children_model_count": m.get("children_model_count"),

        # Dates
        "created_at": m.get("createdAt"),
        "last_modified": m.get("lastModified"),

        # Access & Availability
        "private": m.get("private", False),
        "gated": m.get("gated"),
        "disabled": m.get("disabled", False),
        "inference": m.get("inference"),
        "inference_provider_mapping": m.get("inference_provider_mapping", []),

        # Technical Details
        "config": m.get("config"),
        "safetensors_total_parameters": safetensors.get("total"),
        "safetensors_parameters": safetensors.get("parameters"),
        "transformers_auto_model": transformers_info.get("auto_model"),
        "transformers_pipeline_tag": transformers_info.get("pipeline_tag"),
        "transformers_processor": transformers_info.get("processor"),
        "mask_token": m.get("mask_token"),
        "gguf": m.get("gguf"),

        # Evaluation Results
        "eval_results": m.get("eval_results", []),

        # Related Resources
        "spaces": m.get("spaces", []),
        "siblings": [
            {
                "filename": f.get("rfilename"),
                "size_bytes": f.get("size"),
            }
            for f in (m.get("siblings") or [])
        ],

        # Storage & Security
        "used_storage_bytes": m.get("used_storage"),
        "security_repo_status": m.get("security_repo_status"),

        # Card Metadata
        "card_data": m.get("cardData"),
        "model_index": m.get("model-index"),
    }


def build_dataset_record(d: dict) -> dict:
    """Extract all available fields from a dataset entry."""
    return {
        # Identity
        "id": d.get("id"),
        "author": d.get("author"),
        "sha": d.get("sha"),
        "description": d.get("description"),
        "citation": d.get("citation"),

        # Tags & References
        "tags": d.get("tags", []),
        "paperswithcode_id": d.get("paperswithcode_id"),

        # Engagement Metrics
        "downloads": d.get("downloads", 0) or 0,
        "downloads_all_time": d.get("downloads_all_time", 0) or 0,
        "likes": d.get("likes", 0) or 0,
        "trending_score": d.get("trending_score"),

        # Dates
        "created_at": d.get("createdAt"),
        "last_modified": d.get("lastModified"),

        # Access
        "private": d.get("private", False),
        "gated": d.get("gated"),
        "disabled": d.get("disabled", False),

        # Files
        "siblings": [
            {
                "filename": f.get("rfilename"),
                "size_bytes": f.get("size"),
            }
            for f in (d.get("siblings") or [])
        ],

        # Storage
        "used_storage_bytes": d.get("used_storage"),

        # Card Metadata
        "card_data": d.get("cardData"),
    }


def build_summary(models: list, datasets: list) -> dict:
    tasks: dict = {}
    for m in models:
        task = m.get("pipeline_tag") or "other"
        tasks[task] = tasks.get(task, 0) + 1

    top_model = max(models, key=lambda m: m["downloads"], default=None)
    top_dataset = max(datasets, key=lambda d: d["downloads"], default=None)

    return {
        "total_models": len(models),
        "total_datasets": len(datasets),
        "total_model_downloads": sum(m["downloads"] for m in models),
        "total_model_downloads_all_time": sum(m["downloads_all_time"] for m in models),
        "total_dataset_downloads": sum(d["downloads"] for d in datasets),
        "total_dataset_downloads_all_time": sum(d["downloads_all_time"] for d in datasets),
        "total_model_likes": sum(m["likes"] for m in models),
        "total_dataset_likes": sum(d["likes"] for d in datasets),
        "models_by_task": tasks,
        "top_downloaded_model": top_model["id"] if top_model else None,
        "top_downloaded_model_downloads": top_model["downloads"] if top_model else 0,
        "top_downloaded_dataset": top_dataset["id"] if top_dataset else None,
        "top_downloaded_dataset_downloads": top_dataset["downloads"] if top_dataset else 0,
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
    return "s3://" + bucket + "/" + key


def lambda_handler(event, context):
    """Lambda entry point — extracts data and saves to S3 as JSON."""
    print("Fetching Amazon models from Hugging Face...")
    models = [build_model_record(m) for m in fetch_amazon_models(limit=100)]
    print(f"  Found {len(models)} models.")

    print("Fetching Amazon datasets from Hugging Face...")
    datasets = [build_dataset_record(d) for d in fetch_amazon_datasets(limit=100)]
    print(f"  Found {len(datasets)} datasets.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    s3_key = S3_FOLDER + "/amazon_hf_data_" + timestamp + ".json"

    output = {
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "author": AUTHOR,
        "summary": build_summary(models, datasets),
        "models": models,
        "datasets": datasets,
    }

    print(f"Saving to S3: s3://{S3_BUCKET}/{s3_key}")
    s3_uri = save_to_s3(output, S3_BUCKET, s3_key)
    print(f"Saved successfully: {s3_uri}")

    return {
        "statusCode": 200,
        "message": "Data extracted and saved successfully.",
        "s3_uri": s3_uri,
        "models_count": len(models),
        "datasets_count": len(datasets),
    }
