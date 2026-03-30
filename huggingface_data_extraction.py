"""
Hugging Face API Data Extraction
Extracts models, datasets, and spaces information from the Hugging Face Hub API.
"""

import requests
import json
from typing import Optional


BASE_URL = "https://huggingface.co/api"


def get_models(
    search: Optional[str] = None,
    task: Optional[str] = None,
    limit: int = 10,
    token: Optional[str] = None,
) -> list[dict]:
    """Fetch models from the Hugging Face Hub API."""
    url = f"{BASE_URL}/models"
    params = {"limit": limit}
    if search:
        params["search"] = search
    if task:
        params["pipeline_tag"] = task

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def get_model_info(model_id: str, token: Optional[str] = None) -> dict:
    """Fetch detailed info for a specific model."""
    url = f"{BASE_URL}/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_datasets(
    search: Optional[str] = None,
    limit: int = 10,
    token: Optional[str] = None,
) -> list[dict]:
    """Fetch datasets from the Hugging Face Hub API."""
    url = f"{BASE_URL}/datasets"
    params = {"limit": limit}
    if search:
        params["search"] = search

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def get_dataset_info(dataset_id: str, token: Optional[str] = None) -> dict:
    """Fetch detailed info for a specific dataset."""
    url = f"{BASE_URL}/datasets/{dataset_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_spaces(
    search: Optional[str] = None,
    limit: int = 10,
    token: Optional[str] = None,
) -> list[dict]:
    """Fetch Spaces from the Hugging Face Hub API."""
    url = f"{BASE_URL}/spaces"
    params = {"limit": limit}
    if search:
        params["search"] = search

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def get_user_info(username: str, token: Optional[str] = None) -> dict:
    """Fetch public info for a Hugging Face user or organization."""
    url = f"{BASE_URL}/users/{username}/overview"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def pretty_print(data: dict | list, indent: int = 2) -> None:
    print(json.dumps(data, indent=indent, default=str))


if __name__ == "__main__":
    # Optional: set your HF token for private/gated models
    HF_TOKEN = None  # Replace with your token: "hf_..."

    print("=" * 60)
    print("1. Top 5 text-generation models")
    print("=" * 60)
    models = get_models(task="text-generation", limit=5, token=HF_TOKEN)
    for m in models:
        print(f"  - {m.get('id')}  (downloads: {m.get('downloads', 'N/A')})")

    print("\n" + "=" * 60)
    print("2. Searching models for 'bert'")
    print("=" * 60)
    bert_models = get_models(search="bert", limit=5, token=HF_TOKEN)
    for m in bert_models:
        print(f"  - {m.get('id')}")

    print("\n" + "=" * 60)
    print("3. Model details: bert-base-uncased")
    print("=" * 60)
    model_info = get_model_info("bert-base-uncased", token=HF_TOKEN)
    print(f"  ID       : {model_info.get('id')}")
    print(f"  Author   : {model_info.get('author')}")
    print(f"  Downloads: {model_info.get('downloads', 'N/A')}")
    print(f"  Likes    : {model_info.get('likes', 'N/A')}")
    print(f"  Tags     : {model_info.get('tags', [])[:5]}")

    print("\n" + "=" * 60)
    print("4. Top 5 datasets")
    print("=" * 60)
    datasets = get_datasets(limit=5, token=HF_TOKEN)
    for d in datasets:
        print(f"  - {d.get('id')}  (downloads: {d.get('downloads', 'N/A')})")

    print("\n" + "=" * 60)
    print("5. Dataset details: squad")
    print("=" * 60)
    ds_info = get_dataset_info("rajpurkar/squad", token=HF_TOKEN)
    print(f"  ID       : {ds_info.get('id')}")
    print(f"  Author   : {ds_info.get('author')}")
    print(f"  Downloads: {ds_info.get('downloads', 'N/A')}")
    print(f"  Likes    : {ds_info.get('likes', 'N/A')}")

    print("\n" + "=" * 60)
    print("6. Top 5 Spaces")
    print("=" * 60)
    spaces = get_spaces(limit=5, token=HF_TOKEN)
    for s in spaces:
        print(f"  - {s.get('id')}")
