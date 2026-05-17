import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base
from gra import Gra
# from config import cfg

class SelectiveKernelFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ),
        ])

        hidden = max(out_channels // reduction, 4)
        self.attention_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.attention_heads = nn.ModuleList([
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True)
            for _ in range(len(self.branches))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_features = [branch(x) for branch in self.branches]
        pooled = self.attention_mlp(sum(branch_features))
        attention_logits = torch.stack([head(pooled) for head in self.attention_heads], dim=1)
        attention = F.softmax(attention_logits, dim=1)

        fused = 0
        for idx, feat in enumerate(branch_features):
            fused = fused + feat * attention[:, idx]
        return fused


class SmartRouterFrontEnd(base.MemoryModule):
    """
    智能门控前端：将输入的事件流解耦为三个独立状态特征图。
    状态定义：
    - 瞬态 (Transient): 高 beta, 低 alpha (快速响应，快速遗忘 -> 捕捉运动边缘)
    - 稳态 (Stable): 高 beta, 高 alpha (持续积累 -> 捕捉背景/静态物体)
    - 抑制态 (Suppressive): 低 beta (主动抑制噪声/背景)
    """
    def __init__(self, in_channels=3, hidden_channels=16, state_channels=3, H=128, W=128):
        super().__init__()
        self.feature_proj = SelectiveKernelFusion(in_channels=in_channels, out_channels=state_channels)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, state_channels * 2, kernel_size=1),
        )

        self.state_channels = state_channels
        self.H = H
        self.W = W
        self.register_memory("S_old", torch.zeros(1, state_channels, H, W))

    def _match_memory_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        S_old = self.S_old.to(device=device, dtype=dtype)
        if S_old.shape[0] == batch_size:
            return S_old
        if S_old.shape[0] == 1:
            return S_old.expand(batch_size, -1, -1, -1)
        return torch.zeros(batch_size, self.state_channels, self.H, self.W, device=device, dtype=dtype)

    def forward(self, E_current: torch.Tensor) -> torch.Tensor:
        if E_current.shape[1] != 3:
            raise ValueError(
                f"SmartRouterFrontEnd expects hybrid input with 3 channels, got {E_current.shape[1]}."
            )

        E_features = self.feature_proj(E_current)
        gate_out = self.router(E_current)

        alpha = torch.sigmoid(gate_out[:, 0:self.state_channels])
        beta = F.softmax(gate_out[:, self.state_channels:], dim=1)

        if getattr(self, "fix_alpha", False):
            alpha = alpha.detach()

        S_old = self._match_memory_batch(E_current.shape[0], E_current.device, E_current.dtype)
        S_new = S_old * alpha + E_features * beta
        self.S_old = S_new.detach()
        return S_new
