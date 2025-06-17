import os
import torch
import nibabel as nib

def generate_prediction_mask(pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Generates a multi-class segmentation mask from model logits.
    It thresholds probabilities to create binary masks for each tumor region
    and then combines them into a single 3D label map based on specified priorities.

    Args:
        pred_logits (torch.Tensor): Model output logits, typically after sliding window inference.
                                    Expected shape: (B, C, H, W, D), where C is number of classes.

    Returns:
        torch.Tensor: A 3D tensor representing the combined segmentation mask with integer labels.
                      Shape: (H, W, D).
    """
    output_probs = (torch.sigmoid(pred_logits) > 0.5)
    _, _, H, W, D = output_probs.shape

    output = output_probs[0]
    seg_mask = torch.zeros((H, W, D))

    seg_mask[torch.where(output[1, ...] == 1)] = 2  # WT --> ED
    seg_mask[torch.where(output[2, ...] == 1)] = 1  # TC --> NCR
    seg_mask[torch.where(output[3, ...] == 1)] = 3  # ET --> ET

    return seg_mask.float()


def save_prediction_mask(
    prediction_mask: torch.Tensor,
    metadata: list,
    output_dir: str,
    file_name: str = "prediction_label.nii.gz"
):
    """
    Saves the generated 3D segmentation mask as a NIfTI file.

    Args:
        prediction_mask (torch.Tensor): The 3D tensor of segmentation labels. Shape (H, W, D).
        metadata (list): List of metadata dictionaries (e.g., from original T1c),
                         used to retrieve affine transformation for saving NIfTI.
        output_dir (str): Directory where the NIfTI prediction file will be saved.
        file_name (str): Name of the output NIfTI file.
    """
    os.makedirs(output_dir, exist_ok=True)

    nifti_pred = nib.Nifti1Image(prediction_mask.cpu().numpy(), affine=metadata[0]['affine'])
    nifti_pred.header.set_intent('label', name='Label Map')

    file_path = os.path.join(output_dir, file_name)
    nib.save(nifti_pred, file_path)
    print(f"Saved prediction mask: {file_path}") 