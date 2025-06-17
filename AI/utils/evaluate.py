import torch
from tqdm import tqdm
from losses.loss import SegmentationLoss
from kd_modules.framework import KD_Framework
from utils.logger import log_validation_losses


def process_validation_batch(
    model: KD_Framework,
    batch_data: dict,
    loss_fn: SegmentationLoss,
) -> tuple[float, str]:
    """
    Processes a single batch during validation, computes loss and updates metrics.

    Args:
        model (KD_Framework): The student model wrapper.
        batch_data (dict): A dictionary containing 'imgs', 'masks', and 'data_type' for the batch.
        loss_fn (SegmentationLoss): The segmentation loss function.

    Returns:
        tuple[float, str]: The calculated validation loss for the batch and its data type.
    """
    imgs = batch_data['imgs'].to('cuda:1')
    masks = batch_data['masks'].to('cuda:1')
    data_type = batch_data['data_type'][0]

    with torch.amp.autocast('cuda:1'):
        output = model.student(imgs)
        val_loss = loss_fn(output['pred'], masks)

    print(f"Validation loss per batch: {val_loss}")
    return val_loss, data_type


def validate_epoch(
    model: KD_Framework,
    loader: torch.utils.data.DataLoader,
    epoch: int,
    task,
) -> float:
    """
    Performs a full validation pass over the dataset and logs results.

    Args:
        model (KD_Framework): The student model wrapper.
        loader (DataLoader): DataLoader for the validation dataset.
        epoch (int): Current epoch number (for logging purposes).
        task (clearml.Task): ClearML Task object for reporting metrics.

    Returns:
        float: The average validation loss for the epoch.
    """
    torch.manual_seed(0)
    model.eval()
    loss_fn = SegmentationLoss()
    n_val_batches = len(loader)

    tumors_val_losses = {'GLI': [], 'PED': [], 'SSA': [], 'MEN':[], 'MET':[]}
    running_loss = 0.0

    print(f"Starting validation for epoch {epoch + 1}...")
    with tqdm(total=n_val_batches, desc='Validating', unit='batch', leave=False) as pbar:
        with torch.no_grad():
            for y in loader:
                val_loss, data_type = process_validation_batch(
                    model, y, loss_fn
                )
                tumors_val_losses[data_type].append(val_loss)
                running_loss += val_loss
                pbar.update(1)

    epoch_val_loss = running_loss / n_val_batches
    log_validation_losses(task, tumors_val_losses, epoch, epoch_val_loss)
    print(f"------Final validation loss after epoch {epoch + 1}: {epoch_val_loss:.4f}-------")

    model.train()
    return epoch_val_loss