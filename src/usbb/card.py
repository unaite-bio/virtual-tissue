"""Method cards: what a method takes, gives and needs, read without importing its code."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

METHODS_ROOT = Path(__file__).parent / "methods"
TYPES = {
    str: ("method_id", "class", "label", "summary", "description", "env_group", "intended_usage"),
    dict: ("references", "links", "inputs", "parameters", "resources", "heldout"),
    list: ("units", "outputs"),
    bool: ("control",),
}
LENGTHS = {"label": (1, 50), "summary": (15, 180)}
CHOICES = {
    "family": ("encoder", "joint", "fuser", "translator", "matcher"),
    "fit_mode": ("frozen", "fitted", "transductive"),
}
UNIT_TYPES = ("spot", "pseudospot", "cell")
VALUE_SCALES = ("raw_counts", "raw_intensity", "background_subtracted", "scaled", "arcsinh")
INPUT_KEYS = {"matrix": {"value_scale"}, "image": {"um_per_px", "patch_px"}, "embedding": set()}
WEIGHT_KEYS = {"repo", "revision", "files", "licence", "release_date"}
RESOURCES = {
    "cpu": ("lowcpu", "midcpu", "highcpu"),
    "memory": ("lowmem", "midmem", "highmem", "veryhighmem"),
    "time": ("lowtime", "midtime", "hightime", "veryhightime"),
    "gpu": (None, "gpu"),
}
HELDOUT = ("N", "Y", "?", "unknown")


class CardError(ValueError):
    """A card breaks the card format; the message lists every problem."""


@dataclass(frozen=True)
class Output:
    latent: str
    informed_by: list[str]


@dataclass(frozen=True)
class MethodCard:
    method_id: str
    class_name: str
    label: str
    summary: str
    description: str
    references: dict
    links: dict
    family: str
    control: bool
    fit_mode: str
    units: list[str]
    inputs: dict[str, dict]
    outputs: list[Output]
    parameters: dict
    weights: dict | None
    env_group: str
    resources: dict
    heldout: dict
    intended_usage: str
    path: Path


# The keys of card.yaml are the MethodCard fields, with `class` for class_name and no path.
KEYS = [
    "class" if f.name == "class_name" else f.name for f in fields(MethodCard) if f.name != "path"
]


def _input_ok(spec) -> bool:
    kind = spec.get("kind") if isinstance(spec, dict) else None
    if kind not in INPUT_KEYS or set(spec) != INPUT_KEYS[kind] | {"kind"}:
        return False
    if kind == "matrix":
        return spec["value_scale"] in VALUE_SCALES
    if kind == "image":
        size, um_per_px = spec["patch_px"], spec["um_per_px"]
        return (
            isinstance(size, int)
            and size > 0
            and isinstance(um_per_px, (int, float))
            and um_per_px > 0
        )
    return True


def _outputs_problems(outputs, inputs) -> list:
    if not outputs or not all(
        isinstance(o, dict) and set(o) == {"latent", "informed_by"} for o in outputs
    ):
        return ["outputs must be a non-empty list of {latent, informed_by}"]
    if len({o["latent"] for o in outputs}) < len(outputs):
        return ["outputs must have distinct latent names"]
    return [
        f"informed_by of {o['latent']} must list inputs of the card"
        for o in outputs
        if not (o["informed_by"] and set(o["informed_by"]) <= set(inputs))
    ]


def _weights_ok(weights) -> bool:
    return (
        isinstance(weights, dict)
        and set(weights) == WEIGHT_KEYS
        and re.fullmatch(r"[0-9a-f]{40}", str(weights["revision"])) is not None
        and isinstance(weights["files"], dict)
        and len(weights["files"]) > 0
        and all(re.fullmatch(r"[0-9a-f]{64}", str(sha)) for sha in weights["files"].values())
    )


def _problems(card, folder) -> list:
    """Every way a parsed card breaks the card format."""
    if not isinstance(card, dict):
        return ["the card must be a YAML mapping"]
    problems = [f"missing key {key}" for key in KEYS if key not in card]
    problems += [f"unknown key {key}" for key in card if key not in KEYS]
    for kind, keys in TYPES.items():
        problems += [
            f"{key} must be a {kind.__name__}"
            for key in keys
            if key in card and not isinstance(card[key], kind)
        ]
    if problems:
        return problems
    if card["method_id"] != folder or not re.fullmatch(r"[a-z_][a-z0-9_]{0,49}", folder):
        problems.append(
            f"method_id must be {folder} (its folder), snake_case, at most 50 characters"
        )
    for key, (low, high) in LENGTHS.items():
        if not low <= len(card[key]) <= high:
            problems.append(f"{key} must have {low} to {high} characters")
    for key in ("description", "env_group", "intended_usage"):
        if not card[key].strip():
            problems.append(f"{key} must not be empty")
    if not card["class"].isidentifier():
        problems.append("class must be the name of the class in method.py")
    for key, choices in CHOICES.items():
        if card[key] not in choices:
            problems.append(f"{key} must be one of {choices}")
    if not card["units"] or not set(card["units"]) <= set(UNIT_TYPES):
        problems.append(f"units must be a non-empty list from {UNIT_TYPES}")
    if not card["inputs"] or not all(_input_ok(spec) for spec in card["inputs"].values()):
        problems.append(
            "inputs must map each modality to {kind: matrix, value_scale},"
            " {kind: image, um_per_px, patch_px} or {kind: embedding}"
        )
    problems += _outputs_problems(card["outputs"], card["inputs"])
    if not isinstance(card["references"].get("doi"), list):
        problems.append("references must be {doi: [...]}")
    if set(card["links"]) != {"repository", "documentation"}:
        problems.append("links must be {repository: ..., documentation: ...}")
    if card["weights"] is not None and not _weights_ok(card["weights"]):
        problems.append(
            "weights must be null or {repo, revision, files: {name: sha256}, licence, release_date}"
        )
    for key, choices in RESOURCES.items():
        if card["resources"].get(key, "missing") not in choices:
            problems.append(f"resources.{key} must be one of {choices}")
    if not set(card["heldout"].values()) <= set(HELDOUT):
        problems.append('heldout must map dataset keys to N, Y, "?" or unknown')
    return problems


def load_card(path) -> MethodCard:
    """Read and validate one card.yaml; raise one CardError listing every problem."""
    path = Path(path)
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems = _problems(card, path.parent.name)
    if problems:
        raise CardError(f"{path}:\n- " + "\n- ".join(problems))
    values = {key: value for key, value in card.items() if key != "class"}
    outputs = [Output(o["latent"], o["informed_by"]) for o in card["outputs"]]
    return MethodCard(**dict(values, class_name=card["class"], outputs=outputs, path=path))


def all_cards(root=None) -> list[MethodCard]:
    """Every card under root (default src/usbb/methods); folders without card.yaml are guides."""
    cards, problems = [], []
    for path in sorted(Path(root or METHODS_ROOT).glob("*/card.yaml")):
        try:
            cards.append(load_card(path))
        except CardError as error:
            problems.append(str(error))
    if problems:
        raise CardError("\n".join(problems))
    return cards


def card_by_id(method_id, root=None) -> MethodCard:
    """The card of one method."""
    return load_card(Path(root or METHODS_ROOT) / method_id / "card.yaml")
