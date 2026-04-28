import torch
import torch.nn as nn

import torch.nn.functional as F

from pfe_crossvit_dual.training.model.attentionRollout import get_trainable_heatmap


def write_in_logs(log_file, message, ammend=True):
    if log_file:
        mode = "a" if ammend else "w"
        with open(log_file, mode) as log_f:
            log_f.write(message)


class IoUConstrainedLoss(nn.Module):
    def __init__(
        self,
        model,
        captor,
        lambda_iou: float = 0.1,
        temperature_mult: float = 10.0,
        device="cuda",
    ):
        super().__init__()
        self.model = model
        self.captor = captor
        self.lambda_iou = lambda_iou
        self.temp_mult = temperature_mult
        self.device = device
        self.ce = nn.CrossEntropyLoss()
        self.eps = 1e-6

    def forward(self, logits, label_gt):
        # Classic loss for classification
        loss_cls = self.ce(logits, label_gt)

        # Target binary mask
        target_img = self.captor.current_mask  # [B, 3, H, W]
        target_mask = (torch.abs(target_img).sum(dim=1, keepdim=True) > 0.1).float()
        target_mask = F.interpolate(
            target_mask,
            size=(self.model.img_size[0], self.model.img_size[0]),
            mode="nearest",
        )

        heatmap = get_trainable_heatmap(self.model, self.device)
        b, c, h, w = heatmap.shape

        # Attention, temperature multiplied to enforce the model to change state.
        flat_attn = heatmap.view(b, -1)
        soft_attn = torch.softmax(flat_attn * self.temp_mult, dim=-1).view(b, c, h, w)

        # Intersection and union, should be derivable
        intersection = (soft_attn * target_mask).sum(dim=(1, 2, 3))
        union = (soft_attn + target_mask - (soft_attn * target_mask)).sum(dim=(1, 2, 3))

        # soft iou = derivability
        soft_iou = (intersection + self.eps) / (union + self.eps)

        # loss_iou = 1 - soft_iou.mean()
        # We use logarithm to intensify the loss when the error is large
        loss_iou = -torch.log(soft_iou + self.eps).mean()

        # returns sum with ponderation
        return loss_cls + self.lambda_iou * loss_iou


class MaskCaptor:
    """Temporaly store the mask which is transmitted to the model durinr forward"""

    def __init__(self):
        self.current_mask = None

    def hook_fn(self, module, input, output):
        # input[1] = segmented_image
        self.current_mask = input[1]
