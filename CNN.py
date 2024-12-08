import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as f
# Trying to use for loop to implement 3D convolution.
# 16x112x112 image with 3 channel(R-G-B)

# input_data = torch.randn(3, 16)
# def cal_conv3D(input:Tensor,feature:Tensor,)


def Depthwise(input: Tensor, filter: Tensor, Padding: int) -> Tensor:
    # Do the zero padding
    padding = nn.ZeroPad3d(1)
    input = padding(input)
    # Get the size and change the format to the numpy
    input_size = input.size()
    filter_size = filter.size()
    input = input.numpy()
    filter = filter.numpy()
    init_array = False
    if filter_size[0] != input_size[0]:
        print("Filter channel not equal to input channel!")
        return torch.randn(0)

    # calculate the convolution and save the parameter
    output = []
    for channel in range(0, input_size[0]):
        out_height_arr = []
        for input_depth in range(0, input_size[1] - 2):
            out_width_arr = []
            for input_height in range(0, input_size[2] - 2):
                out_depth_arr = []
                for input_width in range(0, input_size[3]-2):
                    result = 0
                    for filter_depth in range(0, filter_size[1]):
                        for filter_height in range(0, filter_size[2]):
                            for filter_width in range(0, filter_size[3]):
                                result = result + \
                                    input[channel, input_depth+filter_depth, input_height + filter_height, input_width+filter_width] * \
                                    filter[channel, filter_depth,
                                           filter_height, filter_width]
                    out_depth_arr = np.append(
                        out_depth_arr, result)
                if len(out_width_arr) == 0:
                    out_width_arr = np.expand_dims(out_depth_arr, axis=0)
                else:
                    out_width_arr = np.concatenate(
                        [out_width_arr, np.expand_dims(out_depth_arr, axis=0)], axis=0)
            if len(out_height_arr) == 0:
                out_height_arr = np.expand_dims(out_width_arr, axis=0)
            else:
                out_height_arr = np.concatenate(
                    [out_height_arr, np.expand_dims(out_width_arr, axis=0)], axis=0)
        if len(output) == 0:
            output = np.expand_dims(out_height_arr, axis=0)
        else:
            output = np.concatenate([output, np.expand_dims(
                out_height_arr, axis=0)], axis=0)
    return (torch.from_numpy(output))

# input size [a,b,c,d,e]

# if we want to do the pointwise conv we can input filter size : [num,channel,1,1,1]


def conv3D(input: Tensor, weight: Tensor, padding: bool) -> Tensor:
    # Do the padding first
    if (padding):
        padding = nn.ZeroPad3d(1)
        input = padding(input)

    filter_size = weight.size()
    input_size = input.size()
    if filter_size[1] != input_size[0]:
        print("Error input and weight dont have the same channel num")
        return
    output = []
    for num in range(filter_size[0]):
        channel_output = []
        for channel in range(filter_size[1]):
            out_height_arr = []
            for input_z in range(0, input_size[1]):
                out_width_arr = []
                for input_y in range(0, input_size[2]):
                    out_depth_arr = []
                    for input_x in range(0, input_size[3]):
                        result = 0
                        for z in range(filter_size[2]):
                            for y in range(filter_size[3]):
                                for x in range(filter_size[4]):
                                    result = result + input[channel,
                                                            input_z + z, input_y + y, input_x + x] * weight[num, channel, z, y, x]
                                    print(f'filter index: {num},filter channel : {channel}\nInput z-axis : {input_z} , Input y-axis : {
                                        input_y}, Input x-axis : {input_x}\nPadding z-axis : {z},Padding y-axis : {y},Padding x-axis : {x}')
                        out_depth_arr = np.append(
                            out_depth_arr, result)
                    if len(out_width_arr) == 0:
                        out_width_arr = np.expand_dims(out_depth_arr, axis=0)
                    else:
                        out_width_arr = np.concatenate(
                            [out_width_arr, np.expand_dims(out_depth_arr, axis=0)], axis=0)

            # [[3,3,3],[3,3,3]] 3x2
                if len(out_height_arr) == 0:
                    out_height_arr = np.expand_dims(out_width_arr, axis=0)
                else:
                    out_height_arr = np.concatenate(
                        [out_height_arr, np.expand_dims(out_width_arr, axis=0)], axis=0)

            if len(channel_output) == 0:
                channel_output = np.expand_dims(out_height_arr, axis=0)
            else:
                channel_output = np.concatenate(
                    [channel_output, np.expand_dims(out_height_arr, axis=0)], axis=0)
        if len(output) == 0:
            output = np.expand_dims(channel_output, axis=0)
        else:
            output = np.concatenate([output, np.expand_dims(
                channel_output, axis=0)], axis=0)
    return (torch.from_numpy(output))


if __name__ == "__main__":
    # conv3D(input_data, input_weight, input_data)
    # data = Depthwise(input_data, DWC_filter, 1)
    input_data = torch.randn(3, 16, 112, 112)
    input_weight = torch.randn(64, 3, 1, 1, 1)

    DWC_filter = torch.randn(3, 3, 3, 3)

    output = Depthwise(input_data, DWC_filter, 1)
    output = f.max_pool3d(output, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    output = conv3D(output, input_weight, padding=False)
    print("Running...")
    np.save('Output', output.numpy())
    print(output.size())
