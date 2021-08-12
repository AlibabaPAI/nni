import torch

from nni.common.version import TORCH_VERSION

from .literal import QuantDtype


def calculate_qmin_qmax(bits, dtype):
    if dtype == QuantDtype.int:
        qmin, qmax = -2 ** (bits - 1) + 1, 2 ** (bits - 1) - 1
    elif dtype == QuantDtype.uint:
        qmin, qmax = 0, 2 ** bits - 1
    else:
        qmin, qmax = None, None
    return qmin, qmax


def get_bits_length(config, quant_type):
    if isinstance(config["quant_bits"], int):
        return config["quant_bits"]
    else:
        return config["quant_bits"].get(quant_type)


def get_min_max_value(x, target_dim=None):
    if target_dim is None:
        return torch.min(x), torch.max(x)

    indices = list(range(len(x.shape)))
    assert target_dim < len(indices), "target_dim needs to be less than the number of dim of the tensor"
    del indices[target_dim]

    if TORCH_VERSION > (1, 6):
        min_val = torch.amin(x, indices, keepdims=True)
        max_val = torch.amax(x, indices, keepdims=True)
    else:
        min_val = max_val = x
        for ind in indices:
            min_val = torch.min(min_val, dim=ind, keepdim=True)
            max_val = torch.max(max_val, dim=ind, keepdim=True)
    return min_val, max_val
