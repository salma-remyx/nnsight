"""Feature-level model diffing and feature-level control utilities.

Adapted from "Multimodal Model Diffing for Feature Discovery and Control"
(MMDiff, arXiv:2608.09928). Given sparse-autoencoder (SAE) feature
activations from two models on identical inputs - e.g. a base language model
and its multimodal-adapted counterpart - these utilities rank the features
most altered by the fine-tuning (feature isolation), isolate task-specific
features by contrastive firing analysis, and causally remove or steer
individual feature directions inside a trace (feature-level control).

The functions here are pure tensor operations: they run inside or outside a
``model.trace(...)`` context and work with any encoder that maps hidden
states to feature activations (typically an SAE ``encode`` method whose
decoder rows are passed to :func:`steer_features` as directions).
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    """Reshape feature activations ``[..., n_features]`` to ``[N, n_features]``."""
    if features.ndim < 2:
        raise ValueError(
            f"expected feature activations with shape [..., n_features], "
            f"got {tuple(features.shape)}"
        )
    return features.reshape(-1, features.shape[-1]).float()


@dataclass
class FeatureDiff:
    """Per-feature activation comparison between two sets of feature activations.

    Attributes:
        diff (torch.Tensor): ``[n_features]`` mean activation difference
            (target minus source). Positive entries fire more strongly in the
            target model / positive example set.
        source_mean (torch.Tensor): ``[n_features]`` mean activation on the source.
        target_mean (torch.Tensor): ``[n_features]`` mean activation on the target.
        source_rate (torch.Tensor): ``[n_features]`` firing rate (fraction of
            positions with activation > 0) on the source.
        target_rate (torch.Tensor): ``[n_features]`` firing rate on the target.
    """

    diff: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    source_rate: torch.Tensor
    target_rate: torch.Tensor

    def topk(self, k: int, largest: bool = True):
        """Return the ``k`` features with the largest (or most negative) diff.

        Returns:
            A ``torch.return_types.topk`` of per-feature diff values and indices.
        """
        return torch.topk(self.diff, k, largest=largest)


def feature_activation_diff(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
) -> FeatureDiff:
    """Rank features by how much their activations changed between two models.

    MMDiff's "feature isolation" step: encode identical inputs through a
    source model (e.g. a base LM) and a target model (e.g. its
    multimodal-adapted counterpart), then compare mean SAE feature
    activations and firing rates per feature.

    Args:
        source_features: Activations from the source model, ``[..., n_features]``.
        target_features: Activations from the target model, ``[..., n_features]``.
            Must have the same ``n_features`` as ``source_features``.

    Returns:
        A :class:`FeatureDiff` with per-feature mean diff and firing rates.
    """
    source = _flatten_features(source_features)
    target = _flatten_features(target_features)
    if source.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"feature dims differ: source has {source.shape[-1]}, "
            f"target has {target.shape[-1]}"
        )
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    return FeatureDiff(
        diff=target_mean - source_mean,
        source_mean=source_mean,
        target_mean=target_mean,
        source_rate=(source > 0).float().mean(dim=0),
        target_rate=(target > 0).float().mean(dim=0),
    )


def contrastive_feature_scores(
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
) -> FeatureDiff:
    """Isolate task-specific features by contrastive firing analysis.

    MMDiff's "task-specific feature detection" step: compare SAE feature
    activations on a set of positive examples (exhibiting the target
    behavior) against a matched negative set. Features with high positive
    ``diff`` / ``target_rate`` are candidate causal features for the task.

    Args:
        positive_features: Activations on positive examples, ``[..., n_features]``.
        negative_features: Activations on negative examples, ``[..., n_features]``.

    Returns:
        A :class:`FeatureDiff` where ``diff`` / ``target_rate`` describe the
        positive set relative to the negative set.
    """
    return feature_activation_diff(negative_features, positive_features)


def steer_features(
    hidden: torch.Tensor,
    decoder_directions: torch.Tensor,
    indices: Union[int, Sequence[int]],
    alpha: float = 1.0,
    features: Optional[torch.Tensor] = None,
    mode: str = "steer",
) -> torch.Tensor:
    """Causally remove or steer individual SAE feature directions.

    MMDiff's "feature-level control" step, for use inside a trace::

        with model.trace(prompt):
            layer = model.model.language_model.layers[12]
            feats = sae.encode(layer.output)
            layer.output = steer_features(
                layer.output, sae.decoder.weight, [feature_idx],
                features=feats, mode="remove",
            )

    Args:
        hidden: Hidden states to modify, ``[..., d_model]``.
        decoder_directions: SAE decoder rows, ``[n_features, d_model]`` - row
            ``i`` is the direction feature ``i`` writes into the residual stream.
        indices: Feature index or indices to control.
        alpha: Steering strength for ``mode="steer"``.
        features: Feature activations ``[..., n_features]`` aligned with
            ``hidden``. Required for ``mode="remove"``.
        mode: ``"steer"`` adds ``alpha * direction`` at every position;
            ``"remove"`` subtracts each indexed feature's own contribution
            (its activation times its direction), ablating it in SAE space.

    Returns:
        The modified hidden states (a new tensor; assign it back to the
        module's ``.output`` or use ``[:] =`` inside a trace).
    """
    if isinstance(indices, int):
        indices = [indices]
    directions = decoder_directions[list(indices)].to(hidden.dtype)
    if mode == "steer":
        return hidden + alpha * directions.sum(dim=0)
    if mode == "remove":
        if features is None:
            raise ValueError('mode="remove" requires feature activations')
        # Sum_i features[..., i] * direction[i] for the selected features.
        selected = features[..., list(indices)].to(hidden.dtype)
        return hidden - selected @ directions
    raise ValueError(f"unknown mode {mode!r}; expected 'steer' or 'remove'")
