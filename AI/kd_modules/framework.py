import torch
import torch.nn as nn
import torch.nn.functional as F
from kd_modules.cbam_attention import CBAMFeatureExtractor
from models.dyn_unet import DynUNet
from losses.loss import SegmentationLoss

class KD_Framework(nn.Module):
    """
    Custom loss module for Knowledge Distillation (KD) targeting a student segmentation model.
    It combines:
    1. Student's own segmentation loss (Dice + BCE from SegmentationLoss).
    2. KL Divergence loss between attention-enhanced bottleneck feature maps of teacher and student.
    3. Binary Cross-Entropy loss between logits of teacher and student prediction layers (with deep supervision).

    Args:
        temperature (float): Temperature for softening logits in KD. (Default: 5.0)
        alpha (float): Weight for the student's segmentation loss. (Default: 1.0)
        beta (float): Weight for the KL divergence loss on bottleneck features. (Default: 10)
        zeta (float): Weight for the BCE loss on prediction logits. (Default: 0.1)
    """
    def __init__(self, temperature: float = 5.0, alpha: float = 1.0, beta: float = 10.0, zeta: float = 0.1):
        super().__init__()
        self.student = DynUNet(spatial_dims=3, in_channels=4, out_channels=4, deep_supervision=True, KD=True)
        self.loss_fn = SegmentationLoss()
        self.temperature = temperature
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cbam_extractor = CBAMFeatureExtractor(in_channels=512)

        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta

    def forward(self, teacher_outputs: dict, y: dict) -> dict:
        """
        Calculates the total knowledge distillation loss for the student model.

        Args:
            teacher_outputs (dict): Dictionary containing teacher model outputs.
                                    Expected keys: 'pred' (stacked predictions from DS),
                                                   'bottleneck_feature_map'.
            y (dict): Dictionary containing ground truth data.
                      Expected keys: 'imgs' (input images), 'masks' (ground truth segmentation masks).

        Returns:
            dict: A dictionary containing the total student loss and individual weighted loss components.
                  Keys: 'batch_total_student_loss', 'seg_weighted', 'kl_weighted', 'bce_weighted'.
        """
        student_outputs = self.student(y['imgs'])

        # 1. Student's segmentation loss with Deep Supervision (Dice + BCE)
        segloss_s_decoder_1 = self.loss_fn(student_outputs['pred'][:, 0], y['masks'])
        segloss_s_decoder_2 = self.loss_fn(student_outputs['pred'][:, 1], y['masks'])
        segloss_s_decoder_3 = self.loss_fn(student_outputs['pred'][:, 2], y['masks'])
        student_seg_loss = segloss_s_decoder_1 + 0.5 * segloss_s_decoder_2 + 0.25 * segloss_s_decoder_3

        # 2. KD loss between bottleneck layers (KL with CBAM Loss)
        teacher_bottleneck = teacher_outputs['bottleneck_feature_map']
        student_bottleneck = student_outputs['bottleneck_feature_map']

        teacher_bottleneck_att = self.cbam_extractor(teacher_bottleneck)
        student_bottleneck_att = self.cbam_extractor(student_bottleneck)

        student_log_probs = F.log_softmax(student_bottleneck_att / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_bottleneck_att / self.temperature, dim=1)

        kl_loss_with_teacher = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

        # 3. KD loss between prediction layers (BCE Loss on logits with DS)
        bce_loss_with_teacher = 0
        decoder_weights = [1, 0.5, 0.25]

        for decoder_idx, weight in enumerate(decoder_weights):
            teacher_logits = teacher_outputs['pred'][:, decoder_idx] / self.temperature
            student_logits = student_outputs['pred'][:, decoder_idx] / self.temperature

            teacher_hard_labels = [
                (torch.sigmoid(teacher_logits[:, channel_idx]) > 0.5).float().unsqueeze(1)
                for channel_idx in range(1, 4)
            ]

            student_logits_channels = [
                student_logits[:, channel_idx].unsqueeze(1)
                for channel_idx in range(1, 4)
            ]

            bce_losses_per_channel = [
                self.bce_loss(student_logits_channels[i], teacher_hard_labels[i])
                for i in range(len(teacher_hard_labels))
            ]

            bce_loss_with_teacher += sum(bce_losses_per_channel) * (self.temperature ** 2) * weight

        batch_total_student_loss = (self.alpha * student_seg_loss +
                                    self.beta * kl_loss_with_teacher +
                                    self.zeta * bce_loss_with_teacher)

        KD_output = {
            'batch_total_student_loss': batch_total_student_loss,
            'seg_weighted': self.alpha * student_seg_loss,
            'kl_weighted': self.beta * kl_loss_with_teacher,
            'bce_weighted': self.zeta * bce_loss_with_teacher,
        }

        return KD_output 