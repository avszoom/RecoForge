"""Two-tower model: UserTower and ItemTower share a 64-d embedding space.

Architecture (mirrors src/models/README.md):

    User tower input:
        user_id      → Embedding(num_users + 1, 32)   [row 0 = <UNK>]
        activity     → Embedding(3, 8)
        preferred    → Embedding(num_categories, 16) → mean over user's interests
        concat → 56-d → Linear(56, 128) → ReLU → Dropout → Linear(128, 64) → L2-norm

    Item tower input:
        item_id      → Embedding(num_items + 1, 32)   [row 0 = <UNK>]
        category     → Embedding(num_categories, 16)
        text_emb     → frozen MiniLM 384-d, fed in as a fixed feature
        popularity   → scalar
        freshness    → scalar
        concat → 434-d → Linear(434, 256) → ReLU → Dropout → Linear(256, 64) → L2-norm

The score for a (user, item) pair is `dot(user_emb, item_emb)`. Both vectors
are L2-normalized, so the dot product is in [-1, 1]. We multiply scores by
a `logit_scale` factor before the softmax so the loss has a usable range
(scores in [-1, 1] would make softmax nearly uniform, which kills gradients).

Cold start: pass `id = 0` for any user_id or item_id that wasn't in training.
The MLP learns during training to lean on the other features when ID is <UNK>
(see `unk_dropout_rate` in train_two_tower.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TwoTowerConfig:
    num_users: int                 # excluding the <UNK> row
    num_items: int                 # excluding the <UNK> row
    num_categories: int
    num_activity_levels: int = 3
    text_dim: int = 384            # frozen MiniLM output dim

    id_emb_dim: int = 32
    category_emb_dim: int = 16
    activity_emb_dim: int = 8
    output_dim: int = 64

    user_hidden: int = 128
    item_hidden: int = 256
    dropout: float = 0.10

    init_logit_scale: float = math.log(10.0)   # exp(log 10) = 10  → softmax sees scores in roughly [-10, 10]


class UserTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        # +1 to leave row 0 as <UNK>. padding_idx=0 means row 0 stays at zeros
        # and gets no gradient — the MLP learns to handle "no useful ID signal".
        self.user_emb = nn.Embedding(cfg.num_users + 1, cfg.id_emb_dim, padding_idx=0)
        self.activity_emb = nn.Embedding(cfg.num_activity_levels, cfg.activity_emb_dim)
        self.category_emb = nn.Embedding(cfg.num_categories, cfg.category_emb_dim)

        in_dim = cfg.id_emb_dim + cfg.activity_emb_dim + cfg.category_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.user_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.user_hidden, cfg.output_dim),
        )

    def forward(
        self,
        user_id: torch.Tensor,         # (B,) long
        activity: torch.Tensor,        # (B,) long
        preferred_multihot: torch.Tensor,   # (B, num_categories) float, 1.0 where the user has that interest
    ) -> torch.Tensor:
        u = self.user_emb(user_id)                                      # (B, id_dim)
        a = self.activity_emb(activity)                                 # (B, act_dim)
        # Mean of preferred-category embeddings. preferred_multihot @ table = sum, then divide by count.
        cat_sum = preferred_multihot @ self.category_emb.weight         # (B, cat_dim)
        cat_cnt = preferred_multihot.sum(dim=-1, keepdim=True).clamp(min=1.0)
        c = cat_sum / cat_cnt                                           # (B, cat_dim)
        x = torch.cat([u, a, c], dim=-1)                                # (B, in_dim)
        x = self.mlp(x)                                                 # (B, output_dim)
        return F.normalize(x, p=2, dim=-1)                              # L2-normalized


class ItemTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.item_emb = nn.Embedding(cfg.num_items + 1, cfg.id_emb_dim, padding_idx=0)
        self.category_emb = nn.Embedding(cfg.num_categories, cfg.category_emb_dim)

        in_dim = cfg.id_emb_dim + cfg.category_emb_dim + cfg.text_dim + 2   # +2 for popularity, freshness
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.item_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.item_hidden, cfg.output_dim),
        )

    def forward(
        self,
        item_id: torch.Tensor,            # (B,) long
        category: torch.Tensor,           # (B,) long
        text_emb: torch.Tensor,           # (B, text_dim) float — from frozen MiniLM cache
        popularity: torch.Tensor,         # (B,) float in [0, 1]
        freshness: torch.Tensor,          # (B,) float in [0, 1]
    ) -> torch.Tensor:
        i = self.item_emb(item_id)                                      # (B, id_dim)
        c = self.category_emb(category)                                 # (B, cat_dim)
        scalars = torch.stack([popularity, freshness], dim=-1)          # (B, 2)
        x = torch.cat([i, c, text_emb, scalars], dim=-1)                # (B, in_dim)
        x = self.mlp(x)                                                 # (B, output_dim)
        return F.normalize(x, p=2, dim=-1)                              # L2-normalized


class TwoTower(nn.Module):
    """Convenience wrapper: holds both towers and a learnable logit scale."""

    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.user_tower = UserTower(cfg)
        self.item_tower = ItemTower(cfg)
        # Learnable temperature, initialized to log(10). exp() makes it positive.
        self.logit_scale = nn.Parameter(torch.tensor(cfg.init_logit_scale))

    def score_matrix(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute the scaled (B_u, B_i) score matrix.

        With L2-normalized vectors, raw `user_emb @ item_emb.T` lives in [-1, 1].
        We multiply by exp(logit_scale) so softmax has usable dynamic range.
        """
        return (user_emb @ item_emb.T) * self.logit_scale.exp()
