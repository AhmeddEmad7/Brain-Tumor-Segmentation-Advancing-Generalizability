import torch
import torch.nn as nn
from monai.losses import DiceLoss


class SegmentationLoss(nn.Module):
    """
    Combines Dice Loss and Binary Cross-Entropy (BCE) Loss for 3D medical image segmentation.
    This loss function calculates separate Dice+BCE losses for three tumor regions:
    Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET), and sums them up.

    The input `p` (predictions) is expected to be logits, and `y` (ground truth)
    is expected to contain multi-class labels. The loss is computed for binary
    masks derived from these labels.
    """
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.dice = DiceLoss(
            sigmoid=True,
            batch=True,
            smooth_nr=1e-05,
            smooth_dr=1e-05
        )
        self.bce = nn.BCEWithLogitsLoss()

    def _loss(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Helper method to compute the combined Dice and BCE loss for a single tumor region.

        Args:
            p (torch.Tensor): Predicted logits for a specific tumor region (B, 1, D, H, W).
            y (torch.Tensor): Ground truth binary mask for the specific tumor region (B, 1, D, H, W).

        Returns:
            torch.Tensor: The combined Dice and BCE loss for the region.
        """
        return self.dice(p, y) + self.bce(p, y.float())

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the total loss by calculating individual losses for Whole Tumor, Tumor Core,
        and Enhancing Tumor, and summing them.

        Args:
            p (torch.Tensor): Model predictions (logits) of shape (B, C, D, H, W),
                              where C includes background and three tumor regions (e.g., C=4).
                              p[:, 1] for WT, p[:, 2] for TC, p[:, 3] for ET.
            y (torch.Tensor): Ground truth masks of shape (B, 1, D, H, W) with integer labels.
                              Labels are typically: 0 (background), 1 (non-enhancing tumor core),
                              2 (peritumoral edema), 3 (GD-enhancing tumor).

        Returns:
            torch.Tensor: The total combined loss for all tumor regions.
        """

        y_wt = (y > 0).float()
        y_tc = ((y == 1) + (y == 3)).float()
        y_et = (y == 3).float()

        p_wt = p[:, 1].unsqueeze(1)
        p_tc = p[:, 2].unsqueeze(1)
        p_et = p[:, 3].unsqueeze(1)

        l_wt = self._loss(p_wt, y_wt)
        l_tc = self._loss(p_tc, y_tc)
        l_et = self._loss(p_et, y_et)

        return l_wt + l_tc + l_et 