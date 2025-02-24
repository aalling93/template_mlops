import torch
import logging
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import wandb
import os



def train_loop_fn(
    model,
    data_loader,
    optimizer,
    scheduler,
    learning_rate,
    loss_composer,
    weight_decay,
    epochs: int = 2,
    start_epoch:int = 0,
    device: str = "cpu",
    warm_up: int = 0,
    batch_size: int = 32,
    shuffle: bool = True,
    loggertask=None,
    save_file: str = "data/models/best_model",
    save_model: bool = False,
    plot: bool = True,
    notes: str = None,
):
    """
    Function to train a deep learning model using provided data and parameters, with extensive logging and visualization using W&B.

    Sections:
    1. Preparing the data loaders: Initializes data loaders for training, validation, and test sets.
    2. Logging setup: Configures logging and ensures the save directory exists.
    3. Model preparation: Sets the model to training mode and initializes weights.
    4. Training loop: For each epoch, trains and validates the model, logs metrics, and optionally saves checkpoints.
    5. Visualizations and logging: Logs images, performance metrics, and gradient flows to W&B.
    6. Cleanup and preparation for the next epoch: Clears the GPU cache and prepares for the next iteration.

    Args:
        model (torch.nn.Module): The model to be trained.
        data_loader (DataLoader): DataLoader for the dataset, with the dataset attribute.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        learning_rate (float): Initial learning rate.
        loss_composer (function): Function to compute the loss during training.
        weight_decay (float): Weight decay parameter for the optimizer.
        epochs (int): Total number of training epochs. Default is 2.
        device (str): Device to run the training on, e.g., 'cpu' or 'cuda'. Default is 'cpu'.
        warm_up (int): Number of warm-up epochs to gradually increase the learning rate. Default is 0.
        batch_size (int): Batch size for training and validation. Default is 32.
        shuffle (bool): Whether to shuffle the dataset at the beginning of each epoch. Default is True.
        loggertask (object): W&B logging object, typically wandb.run. Default is None.
        save_file (str): Path to save the best model. Default is "data/models/best_model.pth".
        save_model (bool): Whether to save the model when it performs the best. Default is False.
        notes (str): Additional notes or metadata for logging purposes. Default is None.

    Returns:
        None

    Examples:
        >>> model = MyModel()
        >>> data_loader = DataLoader(MyDataset(), batch_size=32, shuffle=True)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        >>> train_loop_fn(model, data_loader, optimizer, scheduler, 0.001, my_loss_fn, 0.0005)

    Note:
        Edge case handling:
        - If epochs is set to 0, the function will skip the training loop entirely.
        - Providing an empty DataLoader will result in early termination during the first iteration check.
        - Extremely large or small learning rates can cause the training to diverge or not progress, respectively.
    """
    # hardcoded paramters.
    generator1 = torch.Generator().manual_seed(42)
    collator = custom_collate_1

    try:
        # Get the root logger
        logger = logging.getLogger()
        if loggertask is not None:
            assert loggertask is wandb.run, "loggertask must be wandb.run"

        ##################################################################################
        # Preparing the data loaders (Train val)
        # etc.
        ##################################################################################
        # just making sure the seed is the same every time.
        
        train_set, val_set, test_set = random_split(data_loader.dataset, [0.70, 0.15, 0.15], generator=generator1)

        

        trainLoader = DataLoader(
            train_set,
            batch_size=data_loader.batch_size,
            shuffle=shuffle,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            persistent_workers=data_loader.persistent_workers,
            prefetch_factor=data_loader.prefetch_factor,
            drop_last=data_loader.drop_last,
            collate_fn=collator,
        )

        valLoader = DataLoader(
            val_set,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=data_loader.num_workers,
            pin_memory=data_loader.pin_memory,
            persistent_workers=data_loader.persistent_workers,
            prefetch_factor=data_loader.prefetch_factor,
            drop_last=data_loader.drop_last,
            collate_fn=collator,
        )

        testLoader = DataLoader(
            test_set,
            batch_size=min(500, len(test_set) - 1),
            shuffle=False,
            collate_fn=collator,
        )
        for x, y, metadata in testLoader:
            # Filter out None values
            test_samples = torch.stack([sample for sample in x if sample is not None])
            test_targets = [torch.stack([target for target in target_list if target is not None]) for target_list in y]
            test_metadata = torch.stack([metadata_list for metadata_list in metadata if metadata_list is not None])

            break

        logger.info(f"Number of samples in training set: {len(train_set)}")
        logger.info(f"Number of samples in validation set: {len(val_set)}")
        logger.info(f"Number of samples in test set: {len(test_samples)}")

        data_loader.dataset.enable_augmentations(False)
        data_loader.dataset.enable_transformations(True)
    except Exception as e:
        pass

        ##################################################################################
        #       Preparing loggin, folder etc.
        #
        ##################################################################################
    try:
        val_losses = []

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        onnx_save_path = os.path.join(os.path.dirname(save_file), "test_model.onnx")
        onnx_save_path = os.path.abspath(onnx_save_path)

        log_dicts = get_log_dicts(
            model,
            trainLoader,
            learning_rate,
            epochs,
            batch_size,
            weight_decay,
            optimizer,
            scheduler,
            loss_composer,
            notes,
        )
        ##################################################################################
        #       Preparing the model
        #
        ##################################################################################
        model.train(True)
        start_lr = optimizer.param_groups[0]["lr"]
        if loggertask is not None:
            wandb.watch(model, None, log="all", log_graph=True)
            wandb.config.update(log_dicts)

        ##################################################################################
        #       TRAINING
        #
        ##################################################################################

    except Exception as e:
        logger.error(f"Error in setup of logging: {e}")
        raise e

    for epoch in range(start_epoch+1, start_epoch+1+epochs):
        pass


    return None
