import torch.nn as nn
import torchvision

class ResNet50TripletHead(nn.Module):
    """
    ResNet-50 backbone (ImageNet pretrained) + 2-layer projection to 128-D + L2-norm.

    If freeze_backbone=True, only trains the projection head.
    """

    def __init__(self, projection_dim=128, freeze_backbone=True):
        super().__init__()
        # 1) Load pretrained ResNet-50
        backbone = torchvision.models.resnet50(pretrained=False)
        # Remove the final fc layer
        backbone.fc = nn.Identity()
        self.backbone = backbone  # outputs (B, 2048)

        # 2) Projection head: 2048 → 512 → projection_dim
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )
        self.norm = nn.LayerNorm(projection_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        returns: (B, projection_dim) L2-normalized
        """
        feat = self.backbone(x)                   # (B, 2048)
        embed = self.projector(feat)              # (B, proj_dim)
        embed = self.norm(embed)                  # (B, proj_dim)
        embed = embed / embed.norm(dim=1, keepdim=True)
        return embed