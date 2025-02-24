import torch
import gc
import psutil

import torch
import torch.nn as nn






def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        torch.mps.empty_cache()
    except:
        pass
    gc.collect()


def print_memory_usage():
    print("Current GPU memory usage: ", torch.cuda.memory_allocated())
    print("Max GPU memory usage: ", torch.cuda.max_memory_allocated())


def get_memory_usage():
    usage = {}
    process = psutil.Process()
    usage["cpu_current"] = process.memory_info().rss / 1024**3  # in GB
    usage["cpu_max"] = psutil.virtual_memory().total / 1024**3  # in GB
    if torch.cuda.is_available():
        usage["cuda_current"] = torch.cuda.memory_allocated() / 1024**3  # in GB
        usage["cuda_max"] = torch.cuda.max_memory_allocated() / 1024**3  # in GB
    else:
        usage["cuda_current"] = 0
        usage["cuda_max"] = 0
    return usage
