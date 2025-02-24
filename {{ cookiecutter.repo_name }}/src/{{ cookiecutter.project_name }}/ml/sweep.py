
import os
import wandb
import subprocess
import logging

"""

It first sets the environment for wandb to operate online and logs in.

It then defines a configuration for the sweep. This includes the method for the sweep (random),
the metric to optimize (Validation loss), and the parameters to vary in the sweep.
Each parameter has a set of values that will be tried in the sweep.

The train function is defined. This function is called for each run of the sweep.
It initializes a run with wandb, converts the configuration parameters for the run into command-line arguments,
and then calls a script named train.py with these arguments using the subprocess.run function.

In the main part of the script, a sweep is created with the defined configuration,
and then the sweep is run with the train function for a specified number of runs (10 in this case).
After all runs are finished, a log message is printed.

This script is a way to automate the process of training a model with different hyperparameters and tracking the results with wandb.
The actual model training is assumed to be implemented in the train.py script, which is not shown here.

"""

# Set Wandb to operate online
os.environ["WANDB_MODE"] = "online"
wandb.login()

# Define the sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "Validation loss", "goal": "minimize"},
    "early_terminate": {"s": 2, "eta": 3, "min_iter": 5, "max_iter": 15},
    "parameters": {
        "LEARNING_RATE": {"values": [0.0005]},  #  "LEARNING_RATE": {"values": np.linspace(0.005, 0.0001, num=1).tolist()},
        "BATCH_SIZE": {"values": [130]},
        "WEIGHT_DECAY": {"values": [0.01]},
        "EPOCHS": {"value": 20},
        "IMAGE_SIZE": {"values": [(2, 128, 128)]},
        "MODEL_WEIGHT_SIZE": {"values": [0.05, 0.1, 0.2, 0.25, 0.5, 0.7, 0.9, 1, 1.25, 1.5, 1.7, 2]},
        "T_0": {"values": [250]},
        "T_MULTI": {"values": [2]},
        "OPTIMIZER_BETAS": {"values": [(0.9, 0.999)]},
        "WARM_UP": {"value": 1},
        "length_thresholds": {"values": [(95, 153), (132, 264), (60, 220)]},
        "OPTIMIZER_NAME": {"values": ["adamw"]},  # {"values": ["adam", "adamw", "adagrad"]}
        "SCHEDULER_NAME": {"values": ["cosine"]},  # , "clyclic"
        "main_path": {"value": "/work3/kaaso/phd/data/sentinel_1/datasets/OpenSARShip2"},
        "WANDB_GROUP": {"values": ["sweeping"]},
        "STRIDE": {"values": [16, 32]},
        "DROPOUT": {"values": [0.05]},
        "FOCAL_LOSS_GAMMA": {"values": [2]},  # "FOCAL_LOSS_GAMMA": {"values": np.linspace(2, 4, num=1).tolist()},
        "FOCAL_LOSS_ALPHA": {"values": [(0.3, 0.9)]},
        "weight_bbox": {"values": [5]},  # "weight_bbox": {"values": np.linspace(10., 10, num=5).tolist()},
        "weight_cls": {"values": [4]},  # "weight_cls": {"values": np.linspace(0.1, 10, num=5).tolist()},
        "weight_size": {"values": [10]},  # "weight_size": {"values": np.linspace(0.1, 10, num=5).tolist()},
        "weight_sog": {"values": [4]},  # "weight_sog": {"values": np.linspace(0.1, 10, num=5).tolist()},
        "weight_cog": {"values": [10]},  # "weight_cog": {"values": np.linspace(0.1, 10, num=5).tolist()},
        "random_rotate_90": {"values": [0.0]},
        "horizontal_flip": {"values": [0.0]},
        "vertical_flip": {"values": [0.0]},
        "PLOT": {"value": True},
        "DEVICE": {"value": "cuda"},
        "NUM_WORKERS": {"value": [4]},
        "crop_size": {"values": [(128, 128)]},
    },
}


def train():
    with wandb.init() as run:
        # Convert sweep parameters to command-line arguments
        args = [
            "python3",
            "train.py",
            "--main_path",
            str(run.config.main_path),
            "--DEVICE",
            str(run.config.DEVICE),
            "--LEARNING_RATE",
            str(run.config.LEARNING_RATE),
            "--BATCH_SIZE",
            str(run.config.BATCH_SIZE),
            "--WEIGHT_DECAY",
            str(run.config.WEIGHT_DECAY),
            "--EPOCHS",
            str(run.config.EPOCHS),
            "--IMAGE_SIZE",
            *map(str, run.config.IMAGE_SIZE),
            "--MODEL_WEIGHT_SIZE",
            str(run.config.MODEL_WEIGHT_SIZE),
            "--T_0",
            str(run.config.T_0),
            "--T_MULTI",
            str(run.config.T_MULTI),
            "--OPTIMIZER_BETAS",
            *map(str, run.config.OPTIMIZER_BETAS),
            "--WARM_UP",
            str(run.config.WARM_UP),
            "--length_thresholds",
            *map(str, run.config.length_thresholds),
            "--OPTIMIZER_NAME",
            str(run.config.OPTIMIZER_NAME),
            "--SCHEDULER_NAME",
            str(run.config.SCHEDULER_NAME),
            "--STRIDE",
            str(run.config.STRIDE),
            "--DROPOUT",
            str(run.config.DROPOUT),
            "--PLOT" if run.config.PLOT else "",
            "--FOCAL_LOSS_GAMMA",
            str(run.config.FOCAL_LOSS_GAMMA),
            "--FOCAL_LOSS_ALPHA",
            *map(str, run.config.FOCAL_LOSS_ALPHA),
            "--weight_bbox",
            str(run.config.weight_bbox),
            "--weight_cls",
            str(run.config.weight_cls),
            "--weight_size",
            str(run.config.weight_size),
            "--weight_sog",
            str(run.config.weight_sog),
            "--weight_cog",
            str(run.config.weight_cog),
            "--random_rotate_90",
            str(run.config.random_rotate_90),
            "--horizontal_flip",
            str(run.config.horizontal_flip),
            "--vertical_flip",
            str(run.config.vertical_flip),
            "--main_path",
            str(run.config.main_path),
            "--DEVICE",
            "cuda" "--WANDB_NAME",
            "SWEEPING BATCH",
            "--PIN_MEMORY",
            "--use_bce_loss",
        ]
        # Call train.py with the specified arguments
        subprocess.run(args)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="sweep")
    wandb.agent(sweep_id, function=train, count=150, project="sweep")
    logging.info("All runs finished. Sweep complete.")
