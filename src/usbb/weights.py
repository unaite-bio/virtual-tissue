"""Pinned Hugging Face weights: download exactly the files a card lists and check every sha256."""

from __future__ import annotations

from pathlib import Path

from usbb.run import file_sha256


def fetch_weights(weights) -> Path:
    """Download a card's weights files at its exact revision; return their folder once all match."""
    from huggingface_hub import snapshot_download  # only cards with weights need it

    folder = Path(
        snapshot_download(
            weights["repo"], revision=weights["revision"], allow_patterns=list(weights["files"])
        )
    )
    wrong = [
        name
        for name, sha256 in weights["files"].items()
        if not (folder / name).is_file() or file_sha256(folder / name) != sha256
    ]
    if wrong:
        raise ValueError(
            f"{weights['repo']} at {weights['revision']}: missing or wrong sha256 for {wrong}"
        )
    return folder
