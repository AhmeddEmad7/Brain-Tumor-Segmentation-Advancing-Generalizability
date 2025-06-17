import torch.optim as optim

def initialize_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:
    """
    Initializes and returns a ReduceLROnPlateau learning rate scheduler.
    This scheduler reduces the learning rate when a metric (e.g., validation loss)
    has stopped improving.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the learning rate will be scheduled.

    Returns:
        torch.optim.lr_scheduler.ReduceLROnPlateau: Configured learning rate scheduler.
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        cooldown=1,
        threshold=0.001,
        min_lr=1e-6
    )
    print("Learning rate scheduler (ReduceLROnPlateau) initialized.")
    return scheduler 