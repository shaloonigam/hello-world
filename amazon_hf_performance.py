"""
Amazon on Hugging Face - Performance Tracker
Fetches all public models and datasets published by Amazon on Hugging Face
and displays their performance metrics (downloads, likes, tasks).

No API token required — all data is public.

Usage:
    python amazon_hf_performance.py
"""

import requests

BASE_URL = "https://huggingface.co/api"
AUTHOR = "amazon"


def fetch_amazon_models(limit: int = 50) -> list[dict]:
    """Fetch all public models published by Amazon."""
    url = f"{BASE_URL}/models"
    params = {"author": AUTHOR, "limit": limit, "sort": "downloads", "direction": -1}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def fetch_amazon_datasets(limit: int = 50) -> list[dict]:
    """Fetch all public datasets published by Amazon."""
    url = f"{BASE_URL}/datasets"
    params = {"author": AUTHOR, "limit": limit, "sort": "downloads", "direction": -1}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def print_models_table(models: list[dict]) -> None:
    print(f"\n{'=' * 75}")
    print(f"  AMAZON MODELS ON HUGGING FACE  (total found: {len(models)})")
    print(f"{'=' * 75}")
    print(f"  {'Model ID':<45} {'Downloads':>10} {'Likes':>6}  Task")
    print(f"  {'-' * 45} {'-' * 10} {'-' * 6}  {'-' * 20}")
    for m in models:
        model_id = m.get("id", "N/A")
        downloads = m.get("downloads", 0) or 0
        likes = m.get("likes", 0) or 0
        task = m.get("pipeline_tag") or "—"
        # Truncate long IDs for display
        display_id = model_id if len(model_id) <= 45 else model_id[:42] + "..."
        print(f"  {display_id:<45} {downloads:>10,} {likes:>6,}  {task}")


def print_datasets_table(datasets: list[dict]) -> None:
    print(f"\n{'=' * 75}")
    print(f"  AMAZON DATASETS ON HUGGING FACE  (total found: {len(datasets)})")
    print(f"{'=' * 75}")
    print(f"  {'Dataset ID':<45} {'Downloads':>10} {'Likes':>6}")
    print(f"  {'-' * 45} {'-' * 10} {'-' * 6}")
    for d in datasets:
        dataset_id = d.get("id", "N/A")
        downloads = d.get("downloads", 0) or 0
        likes = d.get("likes", 0) or 0
        display_id = dataset_id if len(dataset_id) <= 45 else dataset_id[:42] + "..."
        print(f"  {display_id:<45} {downloads:>10,} {likes:>6,}")


def print_summary(models: list[dict], datasets: list[dict]) -> None:
    total_model_downloads = sum(m.get("downloads", 0) or 0 for m in models)
    total_model_likes = sum(m.get("likes", 0) or 0 for m in models)
    total_dataset_downloads = sum(d.get("downloads", 0) or 0 for d in datasets)
    total_dataset_likes = sum(d.get("likes", 0) or 0 for d in datasets)

    # Group models by task
    tasks: dict[str, int] = {}
    for m in models:
        task = m.get("pipeline_tag") or "other"
        tasks[task] = tasks.get(task, 0) + 1
    top_tasks = sorted(tasks.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n{'=' * 75}")
    print("  SUMMARY")
    print(f"{'=' * 75}")
    print(f"  Models   : {len(models):>4}  |  Total Downloads: {total_model_downloads:>12,}  |  Total Likes: {total_model_likes:>6,}")
    print(f"  Datasets : {len(datasets):>4}  |  Total Downloads: {total_dataset_downloads:>12,}  |  Total Likes: {total_dataset_likes:>6,}")
    print(f"\n  Top model tasks:")
    for task, count in top_tasks:
        print(f"    - {task}: {count} model(s)")

    if models:
        top_model = max(models, key=lambda m: m.get("downloads", 0) or 0)
        print(f"\n  Most downloaded model : {top_model.get('id')}  ({top_model.get('downloads', 0):,} downloads)")
    if datasets:
        top_dataset = max(datasets, key=lambda d: d.get("downloads", 0) or 0)
        print(f"  Most downloaded dataset: {top_dataset.get('id')}  ({top_dataset.get('downloads', 0):,} downloads)")
    print()


if __name__ == "__main__":
    print("Fetching Amazon's public models from Hugging Face...")
    models = fetch_amazon_models(limit=50)

    print("Fetching Amazon's public datasets from Hugging Face...")
    datasets = fetch_amazon_datasets(limit=50)

    print_models_table(models)
    print_datasets_table(datasets)
    print_summary(models, datasets)
