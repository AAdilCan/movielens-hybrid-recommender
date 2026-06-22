"""Download and unpack MovieLens releases from GroupLens.

The raw archives are never committed (see ``.gitignore``); instead they are
fetched on demand into ``data/raw``. The download is idempotent: if the
expected files already exist it is skipped.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from recsys.config import RAW_DIR, DatasetConfig, ensure_dirs, get_dataset
from recsys.logging_utils import get_logger

logger = get_logger(__name__)

_CHUNK = 1 << 16  # 64 KiB


def is_downloaded(dataset: DatasetConfig) -> bool:
    """Return True if both the ratings and movies files are already present."""
    ratings = dataset.raw_path / dataset.ratings_file
    movies = dataset.raw_path / dataset.movies_file
    return ratings.is_file() and movies.is_file()


def _download_file(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a progress bar."""
    logger.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with dest.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))


def _extract(zip_path: Path, target: Path) -> None:
    """Extract ``zip_path`` into ``target``."""
    logger.info("Extracting %s", zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)


def download_dataset(name: str = "small", force: bool = False) -> Path:
    """Ensure a MovieLens release is available locally and return its directory.

    Args:
        name: Dataset key from :data:`recsys.config.DATASETS` (``"small"`` or ``"1m"``).
        force: Re-download even if the files already exist.

    Returns:
        Path to the extracted dataset directory (e.g. ``data/raw/ml-latest-small``).
    """
    ensure_dirs()
    dataset = get_dataset(name)

    if is_downloaded(dataset) and not force:
        logger.info("Dataset %r already present at %s", name, dataset.raw_path)
        return dataset.raw_path

    zip_path = RAW_DIR / f"{dataset.archive_subdir}.zip"
    _download_file(dataset.url, zip_path)
    _extract(zip_path, RAW_DIR)
    zip_path.unlink(missing_ok=True)

    if not is_downloaded(dataset):
        raise FileNotFoundError(
            f"Expected {dataset.ratings_file} and {dataset.movies_file} under "
            f"{dataset.raw_path} after extraction, but they are missing."
        )
    logger.info("Dataset %r ready at %s", name, dataset.raw_path)
    return dataset.raw_path
