"""Small comfort regressor designed to run real-time on a Jetson-class GPU.

Backbone: MobileNetV3-Small (timm), ~2.5M params, ~10 ms per 224x224 frame
on a desktop GPU and ~25 ms on Jetson Orin. Single-frame input; the regressor
relies on the fact that the supervision labels are themselves temporally
smoothed, so a per-frame point prediction reproduces a smooth trajectory.

Output:
    score: scalar in [0, 100], trained with Huber loss against `comfort_score`.

Optional auxiliary head:
    soft_5: 5-bin distribution over Likert classes, trained with KL divergence
    when the source provides one (currently only used if a future annotator
    saves soft labels). Disabled by default.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("install timm: pip install timm>=0.9.0") from e


def build_default_model(
    backbone_name: str = "mobilenetv3_small_100",
    pretrained: bool = True,
    aux_soft: bool = False,
) -> "ComfortRegressor":
    return ComfortRegressor(backbone_name=backbone_name, pretrained=pretrained, aux_soft=aux_soft)


class ComfortRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        aux_soft: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat = self.backbone.num_features
        self.score_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.aux_soft = aux_soft
        if aux_soft:
            self.soft_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 5),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        score = self.score_head(feat).squeeze(-1)
        score = torch.sigmoid(score) * 100.0
        out = {"score": score}
        if self.aux_soft:
            out["soft_5"] = self.soft_head(feat)
        return out

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)["score"]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
