from tqdm import tqdm
import torch.nn as nn
import torch
import logging


def train_epoch_fn(model, train_loader, optimizer, loss_composer, device, training: bool = True):
    """
    Trains the model for a single epoch and returns the average loss and logging dictionary.
    Ensures that inputs, targets, and predictions are on the same device as the model and use float32.
    """
    logger = logging.getLogger()

    # Ensure the model is on the correct device with float32
    model = model.to(device, dtype=torch.float32)

    # Disable gradients if not training (inference mode)
    if not training:
        torch.set_grad_enabled(False)

    # Progress bar for batches
    loop = tqdm(train_loader, leave=True)

    # Initialize loss and logging dictionary
    total_loss = []
    logging_dict = {key: 0.0 for key in loss_composer.outputs.keys()}


    return logging_dict




def get_limited_batches(data_loader, num_batches):
    limited_loader = []
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        limited_loader.append(batch)
    return limited_loader
