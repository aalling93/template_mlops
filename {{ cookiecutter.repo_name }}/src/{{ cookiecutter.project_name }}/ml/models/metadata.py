import torch
import gc
import psutil

import torch
import torch.nn as nn

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp




def model_info(model: nn.Module, input_size: tuple) -> dict:
    """
    Returns information about the model including number of parameters, model size, and FLOPs.

    Args:
        model (nn.Module): The PyTorch model.
        input_size (tuple): Input size to be passed to the model (e.g., (1, 3, 224, 224) for a batch size of 1 and an image of 224x224).

    Returns:
        dict: A dictionary containing model statistics.

    # Example usage:
    # model = MyModel()
    # info = model_info(model, input_size=(1, 3, 224, 224))
    # print(info)

    """
    # Detect the device and ensure both model and input are on the same device
    device = next(model.parameters()).device

    # Number of trainable and untrainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable_params = total_params - trainable_params

    # Infer the dtype of the model and determine the size in bytes
    dtype = next(model.parameters()).dtype
    dtype_size_map = {
        torch.float32: 4,  # 4 bytes
        torch.float64: 8,  # 8 bytes
        torch.float16: 2,  # 2 bytes
        torch.int8: 1,     # 1 byte
        torch.int16: 2,    # 2 bytes
        torch.int32: 4,    # 4 bytes
        torch.int64: 8,    # 8 bytes
    }
    param_size_bytes = sum([p.numel() * dtype_size_map[dtype] for p in model.parameters()])
    model_size_mb = param_size_bytes / (1024 * 1024)  # Convert to MB

    # FLOPs estimation by registering hooks
    flops = 0

    def conv_hook(module, input, output):
        nonlocal flops
        output_shape = output.shape
        # Calculate FLOPs for convolutional layers (H_out * W_out * kernel_size * in_channels * out_channels / groups)
        flops += (
            torch.prod(torch.tensor(output_shape[2:])) *  # H_out * W_out (for 2D Conv)
            module.kernel_size[0] * module.kernel_size[1] *  # kernel size
            module.in_channels * module.out_channels // module.groups  # channels / groups
        ).item()

    def linear_hook(module, input, output):
        nonlocal flops
        input_shape = input[0].shape
        # Calculate FLOPs for linear layers (input_size * output_size)
        flops += input_shape[1] * output.shape[1]

    # Register hooks
    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    # Move model and dummy input to the same device
    model.to(device)
    dummy_input = torch.randn(*input_size, dtype=dtype).to(device)

    # Run a forward pass to trigger hooks
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return {
        'trainable_parameters': trainable_params,
        'untrainable_parameters': untrainable_params,
        'model_size_mb': model_size_mb,
        'flops': flops
    }

