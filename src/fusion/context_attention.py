"""
Context-conditioned multimodal attention.

Each modality has its own linear encoder. A scenario id indexes a learned
embedding table. That table is not a language model. Missing modalities are
explicit zero tokens plus a key-padding mask, so attention never reads an
uninitialized tensor.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class ContextConditionedAttention(nn.Module):
    """Shared embedding space with a validity mask."""

    def __init__(
        self,
        input_dims: Optional[Dict[str, int]] = None,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_scenarios: int = 10,
    ):
        super().__init__()
        self.input_dims = input_dims or {"lidar": 64, "camera": 64, "radar": 32, "imu": 16}
        self.modalities = list(self.input_dims)
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, embed_dim) for name, dim in self.input_dims.items()}
        )
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.scenario = nn.Embedding(num_scenarios, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        features: Dict[str, Optional[torch.Tensor]],
        scenario_id: torch.Tensor,
    ):
        """
        Args:
            features: modality name to ``(B, D)`` tensor, or ``None`` if absent.
            scenario_id: ``(B,)`` long tensor.

        Returns:
            fused ``(B, E)``, attention weights ``(B, M, M)``, validity ``(B, M)``.
        """
        batch, device = self._batch_and_device(features)
        tokens = []
        invalid = []
        for name in self.modalities:
            raw = features.get(name)
            if raw is None:
                tokens.append(torch.zeros(batch, self.embed_dim, device=device))
                invalid.append(True)
            else:
                tokens.append(self.projections[name](raw))
                invalid.append(False)
        stacked = torch.stack(tokens, dim=1)
        key_padding = torch.tensor(invalid, device=device).unsqueeze(0).expand(batch, -1)
        if key_padding.all():
            raise ValueError("at least one modality must be present")
        context = self.scenario(scenario_id).unsqueeze(1)
        attended, weights = self.attention(
            stacked + context,
            stacked + context,
            stacked,
            key_padding_mask=key_padding,
        )
        valid = ~key_padding
        weights = weights.masked_fill(key_padding.unsqueeze(1), 0.0)
        pooled = (attended * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        return self.output(pooled), weights, valid

    def _batch_and_device(self, features: Dict[str, Optional[torch.Tensor]]):
        for name in self.modalities:
            raw = features.get(name)
            if raw is not None:
                return raw.shape[0], raw.device
        parameter = next(self.parameters())
        return 1, parameter.device
