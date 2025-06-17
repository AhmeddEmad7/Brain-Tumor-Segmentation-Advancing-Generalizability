import torch
from torch import amp
from tqdm import tqdm
from pathlib import Path
from models.dyn_unet import DynUNet
from kd_modules.framework import KD_Framework
from train_config.optimizer import initialize_optimizer
from train_config.scheduler import initialize_scheduler
from utils.setup_environment import set_seed, create_output_directories
from utils.logger import initialize_clearml_task, log_training_losses, log_learning_rate, log_scalar_metrics
from utils.evaluate import validate_epoch
from torch.utils.data import DataLoader
import argparse


def initialize_models() -> tuple[DynUNet, KD_Framework]:
    """
    Initializes the teacher and student models and moves them to their respective devices.
    Teacher model is on 'cuda:0', Student model and its internal DynUNet are on 'cuda:1'.

    Returns:
        tuple[DynUNet, KD_Framework]: Initialized teacher and student models.
    """
    teacher_model = DynUNet(spatial_dims=3, in_channels=4, out_channels=4, deep_supervision=True, KD=True).to('cuda:0')
    student_model = KD_Framework().to('cuda:1')
    print("Models initialized and moved to respective CUDA devices.")
    return teacher_model, student_model

def load_teacher_model(teacher_model: DynUNet, data_type: str, teacher_model_paths: dict):
    """
    Loads pre-trained weights for the teacher model based on the data type.

    Args:
        teacher_model (DynUNet): The teacher model instance.
        data_type (str): The current data type (e.g., 'GLI', 'SSA').
        teacher_model_paths (dict): Dictionary mapping data types to teacher model checkpoint paths.
    """
    teacher_model_path = teacher_model_paths.get(data_type)
    if teacher_model_path and Path(teacher_model_path).is_file():
        ckpt = torch.load(teacher_model_path, map_location='cuda:0', weights_only=True)
        teacher_model.load_state_dict(ckpt['teacher_model'])
    else:
        print(f"Warning: No teacher model found for data type {data_type} at {teacher_model_path}. Skipping load.")

def load_student_checkpoint(student_model: KD_Framework, optimizer: torch.optim.AdamW,
                            scaler: amp.GradScaler, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, args: argparse.Namespace) -> int:
    """
    Loads a student model checkpoint if available to resume training.

    Args:
        student_model (KD_Framework): The student model.
        optimizer (torch.optim.AdamW): The optimizer.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The learning rate scheduler.
        args (argparse.Namespace): Configuration object containing 'in_checkpoint_dir'.

    Returns:
        int: The epoch to resume training from (0 if no checkpoint found).
    """
    checkpoint_filename = 'Student_model_after_epoch_72_trainLoss_1.5609_valLoss_0.3204.pth'
    checkpoint_path = args.in_checkpoint_dir / checkpoint_filename

    if checkpoint_path.is_file():
        print(f"Found existing student model checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cuda:1', weights_only=False)
        student_model.student.load_state_dict(ckpt['student_model'])
        optimizer.load_state_dict(ckpt['optimizer_student'])
        scaler.load_state_dict(ckpt['grad_scaler_state'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"Loaded student model from checkpoint. Resuming from epoch {ckpt['epoch'] + 1}.")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        return ckpt['epoch'] + 1 # Resume from the next epoch
    print("No student model checkpoint found. Starting training from scratch.")
    return 0

def train_epoch(
    epoch: int,
    trainLoader: DataLoader,
    train_config: dict,
    start_ep: int
) -> tuple[dict, dict]:
    """
    Performs one full training epoch for the student model using knowledge distillation.

    Args:
        epoch (int): Current epoch number (0-indexed).
        trainLoader (DataLoader): DataLoader for the training dataset.
        train_config (dict): Dictionary containing training configurations:
                             'teacher_model', 'student_model', 'optimizer', 'scaler',
                             'accumulation_steps', 'teacher_model_paths', 'task', 'epochs'.
        start_ep (int): The starting epoch number if resuming from a checkpoint.

    Returns:
        tuple[dict, dict]: A dictionary of accumulated average losses for the epoch, and
                           a dictionary of lists of losses per tumor type for the epoch.
    """
    student_model = train_config['student_model']
    teacher_model = train_config['teacher_model']
    optimizer = train_config['optimizer']
    scaler = train_config['scaler']
    accumulation_steps = train_config['accumulation_steps']
    teacher_model_paths = train_config['teacher_model_paths']
    task = train_config['task']
    total_epochs_to_run = train_config['epochs']

    student_model.train()
    teacher_model.eval()

    epoch_losses = {'total': 0.0, 'kl': 0.0, 'seg': 0.0, 'bce': 0.0}
    tumors_losses = {'GLI': [], 'PED': [], 'SSA': [], 'MEN': [], 'MET': []}

    print(f"\nStarting training for epoch {epoch + 1}/{start_ep + total_epochs_to_run}...")
    with tqdm(total=len(trainLoader), desc=f"Epoch {epoch + 1}/{start_ep + total_epochs_to_run}", unit='batch') as pbar:
        optimizer.zero_grad()

        for step, y in enumerate(trainLoader):
            batch_loss = 0.0

            for sub_step in range(len(y['data_type'])):
                imgs_teacher_input = y['imgs'][sub_step].unsqueeze(0).to('cuda:0')
                masks_student_gt = y['masks'][sub_step].unsqueeze(0).to('cuda:1')
                data_type = y['data_type'][sub_step]

                load_teacher_model(teacher_model, data_type, teacher_model_paths)

                with amp.autocast('cuda:0'):
                    teacher_outputs = teacher_model(imgs_teacher_input)

                detached_teacher_output = {k: v.detach().to('cuda:1') for k, v in teacher_outputs.items()}
                imgs_student_input = imgs_teacher_input.to('cuda:1')

                with amp.autocast('cuda:1'):
                    student_outputs = student_model(
                        detached_teacher_output,
                        {'imgs': imgs_student_input, 'masks': masks_student_gt}
                    )
                    loss = student_outputs['batch_total_student_loss'] / accumulation_steps
                    batch_loss += loss.item()
                    tumors_losses[data_type].append(loss.item())

                scaler.scale(loss).backward()

                log_scalar_metrics(
                    task,
                    title=f"Tumors training losses per epoch {epoch+1}",
                    series=f"{data_type} loss",
                    iteration=len(tumors_losses[data_type]),
                    value=float(loss.item())
                )

                for key in epoch_losses:
                    if key != 'total':
                        epoch_losses[key] += (student_outputs.get(f'{key}_weighted', 0) / accumulation_steps)

                if (sub_step + 1) % accumulation_steps == 0 or (sub_step + 1) == len(y['data_type']):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            epoch_losses['total'] += batch_loss
            pbar.update(1)

    for key in epoch_losses:
        epoch_losses[key] /= len(trainLoader)

    return epoch_losses, tumors_losses

def save_checkpoint(
    epoch: int,
    epoch_losses: dict,
    val_loss: float,
    train_config: dict
):
    """
    Saves the model, optimizer, scheduler, and scaler states as a checkpoint.

    Args:
        epoch (int): Current epoch number.
        epoch_losses (dict): Dictionary of average losses from the training epoch.
        val_loss (float): Overall average validation loss for the epoch.
        train_config (dict): Dictionary containing training configurations:
                             'student_model', 'optimizer', 'scheduler', 'scaler',
                             'out_checkpoint_dir'.
    """
    student_model = train_config['student_model']
    optimizer = train_config['optimizer']
    scheduler = train_config['scheduler']
    scaler = train_config['scaler']
    out_checkpoint_dir = train_config['out_checkpoint_dir']

    state = {
        'epoch': epoch,
        'student_model': student_model.student.state_dict(),
        'optimizer_student': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr'],
        'grad_scaler_state': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }

    checkpoint_filename = f'Student_model_after_epoch_{epoch + 1}_trainLoss_{epoch_losses["total"]:.4f}_valLoss_{val_loss:.4f}.pth'
    checkpoint_path = out_checkpoint_dir / checkpoint_filename
    torch.save(state, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")

def run_KD(trainLoader: DataLoader, valLoader: DataLoader, args: argparse.Namespace):
    """
    Main function to run the Knowledge Distillation training process.

    Args:
        trainLoader (DataLoader): DataLoader for the training dataset.
        valLoader (DataLoader): DataLoader for the validation dataset.
        args (argparse.Namespace): Configuration object containing all hyperparameters and paths.
    """
    set_seed(0)
    create_output_directories(args.out_checkpoint_dir)

    teacher_model, student_model = initialize_models()

    optimizer = initialize_optimizer(student_model.parameters(), args)
    scheduler = initialize_scheduler(optimizer)
    scaler = amp.GradScaler('cuda:1')

    teacher_model_paths = args.teacher_model_paths
    start_epoch = load_student_checkpoint(student_model, optimizer, scaler, scheduler, args)

    task = initialize_clearml_task(
        project_name="Fairness KD 5 Tumors Models",
        task_name=f"Fairness KD 5 Tumors with CBAM(KL)+BCE+SEG (Restarted from {start_epoch})",
        args=args,
        tags=['CBAM(KL)+BCE+SEG', 'DynUNet']
    )

    print(f'''Starting Knowledge Distillation Training:
            Total Epochs:    {args.epochs} (Running from {start_epoch + 1} to {start_epoch + args.epochs})
            Batch size:      {args.train_batch_size} (effective through gradient accumulation)
            Learning rate:   {args.learning_rate:.6f}
            Data directories: {args.data_dirs}
    ''')

    train_config = {
        'teacher_model': teacher_model,
        'student_model': student_model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'scaler': scaler,
        'accumulation_steps': args.train_batch_size,
        'teacher_model_paths': teacher_model_paths,
        'out_checkpoint_dir': args.out_checkpoint_dir,
        'task': task,
        'epochs': args.epochs
    }

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_losses, tumors_losses = train_epoch(epoch, trainLoader, train_config, start_epoch)
        log_training_losses(task, epoch_losses, tumors_losses, epoch)

        val_loss = validate_epoch(student_model, valLoader, epoch, task)
        scheduler.step(val_loss)
        log_learning_rate(task, optimizer.param_groups[0]['lr'], epoch + 1)

        save_checkpoint(epoch, epoch_losses, val_loss, train_config)

    print("Training completed.")
    task.close()