import torch.optim as optim
import torch.nn as nn
import argparse


def initialize_optimizer(model_parameters: nn.Parameter, args: argparse.Namespace) -> optim.AdamW:
    """
    Initializes and returns an AdamW optimizer for the given model parameters.

    Args:
        model_parameters (nn.Parameter): Parameters of the model to optimize.
        args (argparse.Namespace): Configuration object containing 'learning_rate' and 'weight_decay'.

    Returns:
        torch.optim.AdamW: Configured AdamW optimizer.
    """
    optimizer = optim.AdamW(
        model_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-4
    )
    print(f"Optimizer (AdamW) initialized with learning rate: {args.learning_rate} and weight decay: {args.weight_decay}")
    return optimizer 