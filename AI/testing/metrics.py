import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import torch.nn as nn

class ComputeMetrics(nn.Module):
    def __init__(self):
        super(ComputeMetrics, self).__init__()
        self.dice_metric = DiceMetric(reduction="mean_batch")
        self.hausdorff_metric = HausdorffDistanceMetric(percentile=95.0, reduction="mean_batch")

    def compute(self, p, y, lbl):
        self.dice_metric.reset()
        self.hausdorff_metric.reset()
        
        print(f"{lbl} - Prediction unique values: {torch.unique(p)}")
        print(f"{lbl} - Ground truth unique values: {torch.unique(y)}")

        if torch.sum(y.float()) == 0 and torch.sum(p.float()) == 0:  # True Negative Case: No foreground pixels in GT
            print(f"{lbl} - No positive samples in ground truth.")
            print(f"Dice scores for {lbl} for this batch: {1.0}")
            print(f"Hausdorff distances for {lbl} for this batch: {0.0}")
            return torch.tensor(1.0), torch.tensor(0.0)
        
        if torch.sum(p.float()) == 0 and torch.sum(y.float()) > 0:  # False Negative Case: GT has 1s, Prediction is all 0s
            print(f"{lbl} - False Negative Case: GT has positive samples, but prediction is empty.")
            print(f"Dice scores for {lbl} for this batch: {0.0}")
            print(f"Hausdorff distances for {lbl} for this batch: {373.1287}")
            return torch.tensor(0.0), torch.tensor(373.1287)
        
        if torch.sum(p.float()) > 0 and torch.sum(y.float()) == 0:  # False Positive Case: Prediction has 1s, GT is all 0s
            print(f"{lbl} - False Positive Case: Prediction has positives, but ground truth is empty.")
            print(f"Dice scores for {lbl} for this batch: {0.0}")
            print(f"Hausdorff distances for {lbl} for this batch: {373.1287}")
            return torch.tensor(0.0), torch.tensor(373.1287)

        dice_score = self.dice_metric(p.float(), y.float())
        hausdorff_dist = self.hausdorff_metric(p.float(), y.float())

        print(f"Dice scores for {lbl} for this batch:\n {dice_score.item()}")
        print(f"Hausdorff distances for {lbl} for this batch:\n{hausdorff_dist.item()}")
    
        return dice_score, hausdorff_dist

    def forward(self, p, y):
        p = (torch.sigmoid(p) > 0.5)
        y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3
        p_wt, p_tc, p_et = p[:, 1].unsqueeze(1), p[:, 2].unsqueeze(1), p[:, 3].unsqueeze(1)
        
        dice_wt, hd_wt = self.compute(p_wt, y_wt, 'wt')
        dice_tc, hd_tc = self.compute(p_tc, y_tc, 'tc')
        dice_et, hd_et = self.compute(p_et, y_et, 'et')
        
        return [dice_wt, hd_wt], [dice_tc, hd_tc], [dice_et, hd_et]