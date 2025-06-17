from clearml import Task
import argparse

def initialize_clearml_task(project_name: str, task_name: str, args: argparse.Namespace, tags: list = None) -> Task:
    """
    Initializes and connects to a ClearML Task.

    Args:
        project_name (str): The name of the ClearML project.
        task_name (str): The name of the specific task.
        args (argparse.Namespace): Configuration object containing all hyperparameters and paths.
        tags (list, optional): List of tags to associate with the task. Defaults to None.

    Returns:
        clearml.Task: The initialized ClearML Task object.
    """
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        reuse_last_task_id=False
    )
    task.connect(dict(args))
    if tags:
        task.add_tags(tags)
    print(f"ClearML Task initialized: Project='{project_name}', Task='{task_name}'")
    return task


def log_scalar_metrics(
    task: Task,
    title: str,
    series: str,
    iteration: int,
    value: float
):
    """
    Reports a scalar metric to ClearML.

    Args:
        task (clearml.Task): The ClearML Task object.
        title (str): The title of the metric chart.
        series (str): The series name within the chart.
        iteration (int): The iteration (e.g., epoch, batch step).
        value (float): The value of the metric.
    """
    task.get_logger().report_scalar(
        title=title,
        series=series,
        iteration=iteration,
        value=value
    )


def log_validation_losses(
    task: Task,
    tumors_val_losses: dict,
    epoch: int,
    epoch_val_loss: float
):
    """
    Logs validation losses to ClearML, both per tumor type and overall.

    Args:
        task (clearml.Task): ClearML Task object.
        tumors_val_losses (dict): Dictionary of lists of losses per tumor type.
        epoch (int): Current epoch number.
        epoch_val_loss (float): Overall average validation loss for the epoch.
    """
    for tumor_type, losses in tumors_val_losses.items():
        if losses:
            avg_loss = sum(losses) / len(losses)
            log_scalar_metrics(
                task,
                title=f"{tumor_type} Losses over Epochs",
                series=f"{tumor_type} Epoch valLoss",
                iteration=epoch + 1,
                value=avg_loss
            )

    log_scalar_metrics(
        task,
        title="KD Train and Val Losses over Epochs",
        series="val loss",
        iteration=epoch + 1,
        value=epoch_val_loss
    )


def log_training_losses(
    task: Task,
    epoch_losses: dict,
    tumors_losses: dict,
    epoch: int
):
    """
    Logs combined Knowledge Distillation training losses over epochs to ClearML.

    Args:
        task (clearml.Task): ClearML Task object.
        epoch_losses (dict): Dictionary of average accumulated losses for the epoch (e.g., total, kl, seg, bce).
        tumors_losses (dict): Dictionary of lists of training losses per tumor type.
        epoch (int): Current epoch number.
    """
    for loss_type, val in epoch_losses.items():
        log_scalar_metrics(
            task,
            title="KD Train and Val Losses over Epochs",
            series=f"{loss_type} loss",
            iteration=epoch + 1,
            value=val
        )

    for tumor_type, losses in tumors_losses.items():
        if losses:
            avg_loss = sum(losses) / len(losses)
            log_scalar_metrics(
                task,
                title=f"{tumor_type} Losses over Epochs",
                series=f"{tumor_type} Epoch trainLoss",
                iteration=epoch + 1,
                value=avg_loss
            )


def log_learning_rate(task: Task, lr: float, iteration: int):
    """
    Logs the current learning rate to ClearML.

    Args:
        task (clearml.Task): ClearML Task object.
        lr (float): Current learning rate.
        iteration (int): Current iteration (e.g., epoch number).
    """
    log_scalar_metrics(
        task,
        title="LR",
        series="learning_rate",
        iteration=iteration,
        value=lr
    )