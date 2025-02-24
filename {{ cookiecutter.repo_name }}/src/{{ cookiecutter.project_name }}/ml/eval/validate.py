import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import onnx

class ModelEvaluator:
    def __init__(self):
        self.model_list = []
        self.dataloader = None
        self.max_length = 458

    def add_model(self, model):
        """Add a model to the model list. Can be a PyTorch model or a path to .onnx or .pt file."""
        if isinstance(model, str):
            if model.endswith('.onnx'):
                # Load ONNX model
                self.model_list.append(onnx.load(model))
            elif model.endswith('.pt'):
                # Load PyTorch model
                self.model_list.append(torch.load(model))
        else:
            self.model_list.append(model)

    def add_dataloader(self, dataloader):
        """Add a dataloader with the data of interest."""
        self.dataloader = dataloader

    def _calculate_metrics(self, prediction, targets):


        metrics_dict = {}



        return metrics_dict

    def metrics(self, to_wandb: bool = True):
        """Calculate metrics and optionally log them to Weights & Biases."""
        metrics_dict = None

        if to_wandb:
            wandb.log(metrics_dict)

        return metrics_dict

    def save_plots(self, save_path):
        """Save the evaluation plots to the specified folder."""
        metrics_dict = self.metrics(to_wandb=False)


    @property
    def model_count(self):
        """Returns the number of models added."""
        return len(self.model_list)

    @property
    def data_size(self):
        """Returns the size of the dataset."""
        return len(self.dataloader.dataset) if self.dataloader else 0
