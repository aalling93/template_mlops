import torch
from torch.optim.lr_scheduler import CyclicLR
import logging
from typing import Optional, List, Union


def get_optimizer(model, optimizer_name="adamw", lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.60), extra_params: Optional[Union[list, List]] = None):
    """
    Create an optimizer for the given model.

    Args:
        model: The model for which to create the optimizer.
        optimizer_name: Name of the optimizer (default: 'adamw').
        lr: Learning rate (default: 0.0001).
        weight_decay: Weight decay factor for regularization (default: 1e-4).
        betas: Coefficients used for computing running averages of gradient (default: (0.9, 0.60)).

    Returns:
        An instance of the specified optimizer.

    Raises:
        AssertionError: If the optimizer name is not in the allowed list or other parameters are out of valid range.
        NotImplementedError: If the optimizer name is not implemented.
    """
    allowed_optimizers = ["adamw", "sgd", "adam", "rmsprop", "adagrad"]

    assert optimizer_name.lower() in allowed_optimizers, f"Optimizer {optimizer_name} not implemented"
    assert 0 <= weight_decay < 1, "Weight decay must be in [0, 1)"
    assert 0 <= betas[0] < 1 and 0 <= betas[1] < 1, "Betas must be in [0, 1)"
    assert lr > -1e-7, "Learning rate must be positive"
    assert lr < 1, "Learning rate must be less than 1"

    parameters = list(model.parameters())
    if extra_params is not None:
        parameters.extend(extra_params)

    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas)

    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay, lr_decay=0.9)

    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")
    return optimizer


def get_scheduler(optimizer, scheduler_name=None, T_0=10, T_mult=2):
    """
    Get the learning rate scheduler based on the specified scheduler name.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is created.
        scheduler_name (str, optional): The name of the scheduler. Defaults to 'cosine'.
        T_0 (int, optional): The number of epochs before the learning rate restarts. Defaults to 10.
        T_mult (int, optional): The factor by which T_0 is multiplied after each restart. Defaults to 2.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.

    Raises:
        AssertionError: If the specified scheduler name is not implemented.
        AssertionError: If T_0 is not positive.
        AssertionError: If T_mult is not positive.
        NotImplementedError: If the specified scheduler name is not implemented.
    """
    allowed_schedulers = ["cosine", "clyclic", None]

    assert scheduler_name in allowed_schedulers, f"Scheduler {scheduler_name} not implemented"
    assert T_0 > 0, "T_0 must be positive"
    assert T_mult > 0, "T_mult must be positive"

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0,
            T_mult,
        )
        return scheduler
    elif scheduler_name == "clyclic":
        scheduler = CyclicLR(
            optimizer,
            base_lr=0.0001,  # lower boundary lr in the cycle for each parameter group
            max_lr=1e-3,  # Upper learning rate boundaries in the cycle for each parameter group
            step_size_up=20,  # Number of training iterations in the increasing half of a cycle
            cycle_momentum=False,
            mode="exp_range",
        )
        return scheduler
    elif scheduler_name is None:
        return None
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")


def get_log_dicts(
    model,
    data_loader,
    learning_rate,
    epochs,
    batch_size,
    weight_decay,
    optimizer,
    scheduler,
    loss_composer,
    Notes: str = None,
):
    logger = logging.getLogger()
    model_sum = {
        #"Parameters": get_n_params(model),
        "Notes": str(Notes),
    }
    # if subsets from random split
    try:
        data_dict = {
            "strides": data_loader.dataset.dataset.strides,
            "main_folder": data_loader.dataset.dataset.main_folder,
            "patch_type": data_loader.dataset.dataset.patch_type,
            "img_type": data_loader.dataset.dataset.img_type,
            "length of output": data_loader.dataset.dataset.output_channels,
            "image transforms": data_loader.dataset.dataset.image_transformations,
            "target transforms": data_loader.dataset.dataset.target_transformations,
            "training data augmentation": data_loader.dataset.dataset.augmentations,
            "training samples length": len(data_loader.dataset),
            "testing samples length": len(data_loader.dataset),
        }
        # otherwise, if not a data subset from pytorch, e.g., using the full dataloader
    except Exception as e:
        logger.warning(f"{e}\nCould not get info from data_loader.dataset.dataset. Trying to get it from data_loader.dataset")
        data_dict = {
            "strides": data_loader.dataset.strides,
            "main_folder": data_loader.dataset.main_folder,
            "patch_type": data_loader.dataset.patch_type,
            "img_type": data_loader.dataset.img_type,
            "length of output": data_loader.dataset.output_channels,
            "image transforms": data_loader.dataset.image_transformations,
            "target transforms": data_loader.dataset.target_transformations,
            "training data augmentation": data_loader.dataset.augmentations,
            "training samples length": len(data_loader.dataset),
            "testing samples length": len(data_loader.dataset),
        }

    hyper_parms = {
        "learning rate": learning_rate,
        "batch size": batch_size,
        "weight decay": weight_decay,
        "epochs": epochs,
    }

    training_parms = {
        "optimizer": f"\n{repr(optimizer)}",
        "scheduler": f"\n{repr(scheduler)}",
        "loss": str(loss_composer),
    }

    log_dicts = {
        "Model": model_sum,
        "Data": data_dict,
        "Hyperparameters": hyper_parms,
        "Training parameters": training_parms,
    }
    return log_dicts


def get_warmupdown(
    start_lr: float = 0.1,
    end_lr: float = 0.001,
    max_epoch: int = 5,
    current_epoch: int = 1,
):
    steps = (end_lr - start_lr) / max_epoch
    lr = steps * current_epoch
    return lr
