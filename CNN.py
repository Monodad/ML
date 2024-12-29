import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as f
from Quantized import quantized_int8
from pathlib import Path
# Trying to use for loop to implement 3D convolution.
# 16x112x112 image with 3 channel(R-G-B)

# input_data = torch.randn(3, 16)
# def cal_conv3D(input:Tensor,feature:Tensor,)


def accumulation(input: Tensor, layer: int):
    return torch.sum(input, dim=layer)


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
    try:
        input_data = torch.from_numpy(np.load('input_data.npy'))
        PWC_filter_1 = torch.from_numpy(np.load('PWC_filter_1a.npy'))
        DWC_filter_1 = torch.from_numpy(np.load('DWC_filter_1a.npy'))
    except:
        input_data = torch.randn(3, 16, 112, 112)
        PWC_filter_1 = torch.randn(64, 3, 1, 1, 1)
        DWC_filter_1 = torch.randn(3, 3, 3, 3)
        input_data = quantized_int8(input_data.float())
        np.save('input_data', input_data.numpy())
        np.save('PWC_filter_1a', quantized_int8(PWC_filter_1.float()).numpy())
        np.save('DWC_filter_1a', quantized_int8(DWC_filter_1.float()).numpy())

    print("Running..")
    # layer 1 input size:3x16x112x112
    try:
        output = torch.from_numpy(np.load('layer1_out.npy'))
    except:
        output = Depthwise(input_data, DWC_filter_1, 1)
        output = f.max_pool3d(output, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        output = conv3D(output, PWC_filter_1, padding=False)
        output = accumulation(output, 1)
        np.save('layer1_out', output.numpy())
        output = quantized_int8(output.float())
        np.save('PWC_1a_int_out', output.numpy())

    # layer 2 input size:64, 16, 56, 56
    try:
        output = torch.from_numpy(np.load('layer2_out.npy'))
        DWC_filter_2 = torch.from_numpy(np.load('DWC_filter_2a.npy'))
        PWC_filter_2 = torch.from_numpy(np.load('PWC_filter_2a.npy'))

    except:
        DWC_filter_2 = torch.randn(64, 3, 3, 3)
        PWC_filter_2 = torch.randn(128, 64, 1, 1, 1)
        np.save('PWC_filter_2a', quantized_int8(PWC_filter_2.float()).numpy())
        np.save('DWC_filter_2a', quantized_int8(DWC_filter_2.float()).numpy())
        output = Depthwise(output, DWC_filter_2, 1)
        output = f.max_pool3d(
            output, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        output = conv3D(output, PWC_filter_2, padding=False)
        output = accumulation(output, 1)
        np.save('layer2_out', output.numpy())
        output = quantized_int8(output.float())
        np.save('PWC_2a_int_out', output.numpy())
    # layer 3 input size: 128 8 28 28
    try:
        output = torch.from_numpy(np.load('layer3_out.npy'))
        DWC_filter_3 = torch.from_numpy(np.load('DWC_filter_3a.npy'))
        PWC_filter_3 = torch.from_numpy(np.load('PWC_filter_3a.npy'))

    except:
        DWC_filter_3 = torch.randn(128, 3, 3, 3)
        PWC_filter_3 = torch.randn(256, 128, 1, 1, 1)
        np.save('PWC_filter_3a', quantized_int8(PWC_filter_3.float()).numpy())
        np.save('DWC_filter_3a', quantized_int8(DWC_filter_3.float()).numpy())
        output = Depthwise(output, DWC_filter_3, 1)
        output = conv3D(output, PWC_filter_3, padding=False)
        output = accumulation(output, 1)
        np.save('layer3_out', output.numpy())
        # output = Depthwise(output, DWC_filter_1, 1)
        output = quantized_int8(output.float())
        np.save('PWC_3a_int_out', output.numpy())

    # layer 3b input size: 256 8 28 28

    try:
        DWC_filter_3 = torch.from_numpy(np.load('DWC_filter_3b.npy'))
        PWC_filter_3 = torch.from_numpy(np.load('PWC_filter_3b.npy'))
        output = torch.from_numpy(np.load('PWC_3b_out.npy'))
    except:
        DWC_filter_3 = torch.randn(256, 3, 3, 3)
        PWC_filter_3 = torch.randn(256, 256, 1, 1, 1)
        output = torch.from_numpy(np.load('PWC_3a_int_out.npy'))
        np.save('PWC_filter_3b', quantized_int8(DWC_filter_3.float()).numpy())
        np.save('DWC_filter_3b', quantized_int8(PWC_filter_3.float()).numpy())
        output = Depthwise(output, DWC_filter_3, 1)
        output = f.max_pool3d(
            output, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        output = conv3D(output, PWC_filter_3, padding=False)
        output = accumulation(output, 1)
        np.save('PWC_3b_out', output.numpy())
        # output = Depthwise(output, DWC_filter_1, 1)
        print(output.size())
        output = torch.from_numpy(np.load('PWC_3b_out.npy'))
        output = quantized_int8(output.float())
        np.save('PWC_3b_int_out', output.numpy())
    # Layer 3b
    # try:
    #     DWC_filter_3 = torch.from_numpy(np.load('DWC_filter_3b.npy'))
    #     PWC_filter_3 = torch.from_numpy(np.load('PWC_filter_3b.npy'))
    #     output = torch.from_numpy(np.load('PWC_3b_out.npy'))
    # except:
    #     DWC_filter_3 = torch.randn(256, 3, 3, 3)
    #     PWC_filter_3 = torch.randn(256, 256, 1, 1, 1)
    #     output = torch.from_numpy(np.load('layer3_out.npy'))
    #     np.save('PWC_filter_3b', PWC_filter_3.numpy())
    #     np.save('DWC_filter_3b', DWC_filter_3.numpy())
    #     output = Depthwise(output, DWC_filter_3, 1)
    #     output = f.max_pool3d(
    #         output, kernel_size=(2, 2, 2), stride=(2, 2, 2))
    #     output = conv3D(output, PWC_filter_3, padding=False)
    #     output = accumulation(output, 1)
    #     np.save('PWC_3b_out', output.numpy())
    #     # output = Depthwise(output, DWC_filter_1, 1)
    #     print(output.size())
    #     output = torch.from_numpy(np.load('PWC_3b_out.npy'))
    #     output = quantized_int8(output.float())
    #     np.save('PWC_3b_int_out', output.numpy())

    # layer DWC PWC 4a
    try:
        output = torch.from_numpy(np.load('PWC_4a_out.npy'))

    except:
        output = torch.from_numpy(np.load('PWC_3b_int_out.npy'))
        DWC_filter_4a = torch.randn(256, 3, 3, 3)
        PWC_filter_4a = torch.randn(512, 256, 1, 1, 1)
        np.save('PWC_filter_4a', quantized_int8(PWC_filter_4a.float()).numpy())
        np.save('DWC_filter_4a', quantized_int8(DWC_filter_4a.float()).numpy())
        output = Depthwise(output, DWC_filter_4a, 1)
        output = conv3D(output, PWC_filter_4a, padding=False)
        output = accumulation(output, 1)
        np.save('PWC_4a_out', output.numpy())
        np.save('PWC_4a_int_out', quantized_int8(output.float()).numpy())
        # output = Depthwise(output, DWC_filter_1, 1)
        print(output.size())
    # Layer DWC PWC 4b
    try:
        output = torch.from_numpy(np.load('PWC_4b_int_out.npy'))

    except:
        output = torch.from_numpy(np.load('PWC_4a_int_out.npy'))
        DWC_filter_4b = torch.randn(512, 3, 3, 3)
        PWC_filter_4b = torch.randn(512, 512, 1, 1, 1)
        np.save('PWC_filter_4b', quantized_int8(PWC_filter_4b.float()).numpy())
        np.save('DWC_filter_4b', quantized_int8(DWC_filter_4b.float()).numpy())
        output = Depthwise(output, DWC_filter_4b, 1)
        output = f.max_pool3d(
            output, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        output = conv3D(output, PWC_filter_4b, padding=False)
        output = accumulation(output, 1)
        np.save('PWC_4b_out', output.numpy())
        np.save('PWC_4b_int_out', quantized_int8(output.float()).numpy())
        # output = Depthwise(output, DWC_filter_1, 1)
        print(output.size())

   # Layer DWC PWC 5a
    try:
        output = torch.from_numpy(np.load('PWC_5a_int_out.npy'))

    except:
        output = torch.from_numpy(np.load('PWC_4b_int_out.npy'))
        DWC_filter_5a = torch.randn(512, 3, 3, 3)
        PWC_filter_5a = torch.randn(512, 512, 1, 1, 1)
        np.save('PWC_filter_5a', quantized_int8(PWC_filter_5a.float()).numpy())
        np.save('DWC_filter_5a', quantized_int8(DWC_filter_5a.float()).numpy())
        output = Depthwise(output, DWC_filter_5a, 1)
        output = conv3D(output, PWC_filter_5a, padding=False)
        output = accumulation(output, 1)
        np.save('PWC_5a_out', output.numpy())
        np.save('PWC_5a_int_out', quantized_int8(output.float()).numpy())
        # output = Depthwise(output, DWC_filter_1, 1)
        print(output.size())

    # layer DWC PWC 5b
    try:
        output = torch.from_numpy(np.load('PWC_5b_int_out.npy'))

    except:
        output = torch.from_numpy(np.load('PWC_5a_int_out.npy'))
        DWC_filter_5b = torch.randn(512, 3, 3, 3)
        PWC_filter_5b = torch.randn(512, 512, 1, 1, 1)
        np.save('PWC_filter_5b', quantized_int8(PWC_filter_5b.float()).numpy())
        np.save('DWC_filter_5b', quantized_int8(DWC_filter_5b.float()).numpy())
        output = Depthwise(output, DWC_filter_5b, 1)
        output = f.max_pool3d(
            output, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        output = conv3D(output, PWC_filter_5b, padding=False)
        output = accumulation(output, 1)
        np.save('PWC_5b_out', output.numpy())
        np.save('PWC_5b_int_out', quantized_int8(output.float()).numpy())
        # output = Depthwise(output, DWC_filter_1, 1)
        print(output.size())
