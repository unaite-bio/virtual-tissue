"""The class every method subclasses, and loading a method from its card."""

from __future__ import annotations

import importlib


class Method:
    """One benchmarked method; a subclass overrides what it needs and the runner does the rest."""

    def __init__(self, card, params, seed, model_dir=None):
        self.card = card
        self.params = params
        self.seed = seed
        self.model_dir = model_dir

    def load(self):
        """Load weights or other resources; runs after the input checks and the cache check."""

    def check(self, inp) -> list:
        """Reasons to refuse inp beyond what the card says; runs before load."""
        return []

    def fit(self, inp):
        """Learn from inp: the train bundle (fitted) or the test bundle (transductive)."""
        raise NotImplementedError(
            f"{type(self).__name__} has fit_mode {self.card.fit_mode} but no fit()"
        )

    def transform(self, inp) -> dict:
        """Return {latent: float32 array (n_units, d)} with rows in inp.unit_ids order."""
        raise NotImplementedError(f"{type(self).__name__} has no transform()")

    def state(self) -> dict:
        """Arrays (name -> array) that must not change during a frozen run."""
        return {}


def load_method(card, params, seed, model_dir=None) -> Method:
    """Build the method of a card with params over the card's defaults; unknown params raise."""
    params = params or {}
    unknown = sorted(set(params) - set(card.parameters))
    if unknown:
        raise ValueError(
            f"{card.method_id} has no parameters {unknown}; it declares {sorted(card.parameters)}"
        )
    module = importlib.import_module(f"usbb.methods.{card.method_id}.method")
    method_class = getattr(module, card.class_name)
    return method_class(card, dict(card.parameters, **params), seed, model_dir)
