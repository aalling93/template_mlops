
import torch
import numpy as np
from typing import Union, List


def test_data(x: torch.Tensor) -> torch.Tensor:
    """ "
    Convert np array to tensor
    """
    if len(x) > 10:
        print(f"Warning: does not have correct channels: {x.shape}")

    # if min(torch.min(x[0]).item(), torch.min(x[1]).item()) < 0:
    #    print(x.shape)
    #    print(
    #        "Warning: Input array has value less than 0:",
    #        min(torch.min(x[0]).item(), torch.min(x[1]).item()),
    #    )
    # if max(torch.max(x[0]).item(), torch.max(x[1]).max()) > 1:
    #    print(
    ##        "Warning: Input array has value greater than 1: ",
    #        max(torch.max(x[0]).item(), torch.max(x[1]).item()),
    #    )
    return x


################################## PIXEL TRANSFORMS ##################################
def add_channel(
    x: np.ndarray,
) -> np.ndarray:
    """ "
    Scale the input to the range [0,1] for each channel
    Assuming that the max and min are found from the CONSTANT list.


    """
    c, h, w = x.shape

    if c == 1:
        # return np.dstack((x, x)).astype(np.float32)
        return np.vstack((x, x))
    else:
        return x


def format_shape(x: np.ndarray) -> np.ndarray:
    """
    make sure the shape is band x h x w

    sometimes, it is h x band x w. Change this..


    
    Band always has fewwse dimentsions, use this 

    i.e. if the shape is (89 , 2, 89) change it to (2, 89, 89)
    if the shape is (2, 89, 89) keep it as it is
    if the shape is (89, 89, 2) change it to (2, 89, 89)



    """
    # find the index of the smallest dimension
    idx = np.argmin(x.shape)
    # if the smallest dimension is the first, then the shape is correct
    if idx == 0:
        return x
    # otherwise, change the shape
    elif idx == 1:
        return np.moveaxis(x, 0, -1)
    else:
        return np.moveaxis(x, idx, 0)
    


def min_max_scale(
    x: np.ndarray,
    data_max=20 * np.log10(65535 + 1),
    data_min=1,
    to_max=1,
    to_min=1e-14,
) -> np.ndarray:
    """ "
    Scale the input to the range [0,1] for each channel
    Assuming that the max and min are found from the CONSTANT list.


    """
    x = np.clip(x, a_min=data_min, a_max=data_max)
    return (to_max - to_min) * ((x - data_min) / ((data_max - data_min)) + to_min + 1e-6)


def scale_shipsize(
    x: np.ndarray,
) -> np.ndarray:
    """ "
    Scale the input to the range [0,1] for each channel
    Assuming that the max and min are found from the CONSTANT list.

    """

    # Seawise Giant is 458m long and 69m wide (the longest ship in the world)
    # Pioneering Spirit is 382m long and 124m wide (the widest ship in the world)
    width = x[..., WIDTH_START, :, :]
    length = x[..., LENGTH_START, :, :]

    length = length / MAX_SHIP_LENGTH
    width = width / MAX_SHIP_WIDTH

    x[..., WIDTH_START, :, :] = width
    x[..., LENGTH_START, :, :] = length

    return x


def scale_sog(x: Union[np.ndarray, list, List, torch.tensor], max_speed: float = MAX_SPEED) -> np.ndarray:
    """ "
    Scale the input to the range [0,1] for each channel
    Assuming that the max and min are found from the CONSTANT list.

    """

    # The highest sog possible is 102.3. This only occurs when there is an error (or helicopter).
    # assuming highest sog is 70 kn.
    # 70 kn = 130 km/h
    # ([11, 32, 32])
    if type(x) is list:
        x_temp = []
        for i in range(len(x)):
            sog = x[i][..., SOG_START, :, :]
            sog = np.clip(sog, a_min=0, a_max=max_speed)
            sog = sog / max_speed
            x_temp.append(sog)
        return x_temp
    else:

        sog = x[..., SOG_START, :, :]
        sog = np.clip(sog, a_min=0, a_max=max_speed)
        sog = sog / max_speed
        x[..., SOG_START, :, :] = sog
        return x


def scale_cog(x: np.ndarray) -> np.ndarray:
    """ "
    Scale the input to the range [0,1] for each channel
    Assuming that the max and min are found from the CONSTANT list.

    """
    # logger = logging.getLogger(__name__)
    # logger.warning("Are you sure your cog is not already between -1 and 1.")
    cog = x[..., COG_START, :, :]
    # cog = np.sin(sog)
    # sog = sog / np.sin(np.deg2rad(360))
    cog = np.clip(cog, a_min=0, a_max=360)
    # cog = cog/360

    x[..., COG_START, :, :] = cog
    return x


def absolute(x: np.ndarray) -> np.ndarray:
    return abs(x + 1e-6)


def to_db(x: np.ndarray) -> np.ndarray:
    x = abs(x) + 1
    return 10 * np.log10(x + 1) + 1e-6


def to_db_inv(x: np.ndarray) -> np.ndarray:
    return 10 ** (x / 10)


# @dispatch(np.ndarray)
def to_db_amplitude(x: np.ndarray) -> np.ndarray:
    x = abs(x) + 1
    return 20 * np.log10(x + 1) + 1e-6


def to_db_amplitude_inv(x: np.ndarray) -> np.ndarray:
    return 10 ** (x / 20) + 1e-6


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Convert an input to a PyTorch tensor without unnecessary copying or warnings.

    Args:
        x (np.ndarray, torch.Tensor, or other): Input data to convert to tensor.

    Returns:
        torch.Tensor: A PyTorch tensor representation of 'x'.
    """
    # If the input is a NumPy array, use torch.from_numpy to avoid copying
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    # If the input is already a PyTorch tensor, return it directly
    elif isinstance(x, torch.Tensor):
        tensor = x
    # For other types, convert to a tensor; this may involve copying
    else:
        tensor = torch.tensor(x, dtype=torch.get_default_dtype())

    return tensor


def to_int8(x):
    """
    Convert to tensor of type int8
    """
    return x.type(torch.int)


def to_quint8(x):
    """
    Convert to tensor of type quint8
    """
    return x.type(torch.quint8)


def to_float32(x):
    """
    Convert to tensor of type float32
    """
    return x.to(torch.float32)  # x.type(torch.float32)


def np2float32(x):
    """
    Convert to tensor of type float32
    """
    return x.astype(np.float32)


def np2float64(x):
    """
    Convert to tensor of type float32
    """
    return x.astype(np.float64)


def np2int8(x):
    """
    Convert to tensor of type float32
    """
    return x.astype(np.int8)


def to_float64(x):
    """ "
    Convert to tensor of type float64
    """
    return x.type(torch.float64)


def min_max_scale_adaptive(
    x: np.ndarray,
) -> np.ndarray:
    """ ""
    Scale the input to the range [0,1] for each channel

    Each individual image will be scaled to [0,1], i.e., there is no global scaling
    """
    x = abs(x)

    taeller = x - np.min(x, (0, 1))
    naevner = np.max(x, (0, 1)) - np.min(x, (0, 1))
    img = taeller / (naevner + 1e-16)
    img = np.clip(img, a_min=1e-6, a_max=1.00)

    return img


def min_max_scale_inv(
    x: np.ndarray,
    data_max=255,
    data_min=0,
    to_max=1,
    to_min=0,
):
    """ "
    Inverse of min_max_scale
    """
    return (x - to_min) / ((to_max - to_min) * (data_max - data_min) + data_min + 1e-6)


def print_stats(x, msg=""):
    print(f"{msg} Min: {np.min(x)}, Max: {np.max(x)}")
