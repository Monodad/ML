import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

# torch.quantize_per_tensor


def quantized_int8(data: Tensor) -> Tensor:
    data_numpy = data.numpy()
    x_max = np.max(np.absolute(data_numpy))
    scale = x_max / 127.0
    return torch.quantize_per_tensor(
        data, scale=scale, zero_point=0, dtype=torch.qint8).int_repr()

    # test
if __name__ == '__main__':
    DWC_filter_3 = torch.randn(256, 3)

    print(np.max(DWC_filter_3.numpy()))
    out = quantized_int8(DWC_filter_3)
    print(out)
    print(np.max(out.numpy()))
    print(np.min(out.numpy()))
    pass
