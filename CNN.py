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
                    # print(out_depth_arr)
                    # [3,3,3

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
        if len(output) == 0:
            output = np.expand_dims(out_height_arr, axis=0)
        else:
            output = np.concatenate([output, np.expand_dims(
                out_height_arr, axis=0)], axis=0)
    return (torch.from_numpy(output))


def conv3D(input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    # Do the padding first
    padding = nn.ZeroPad3d(1)
    input = padding(input)
    # Weight : Flatten
    # print(weight.dim())
    # weight = torch.flatten(
    #     weight, start_dim=weight.dim()-3, end_dim=(weight.dim()-1))
    filter_num = weight.size()[0]
    filter_channels = weight.size()[1]
    input = input.numpy()
    weight = weight.numpy()

    output = []
    filter_out = 0
    print(filter_channels)
    for i in range(filter_channels):
        print(i)
    # with open('test.txt', 'w', encoding='utf-8') as f:
    #     f.write(str(input))
    np.save('test', input)
    arr_loaded = np.load('test.npy')
    print(arr_loaded)
    with open('model_random.txt', 'w', encoding='utf-8') as f:
        for filter in range(filter_num):
            array_filter_channels = []
            for channel in range(filter_channels):
                array_input_z = []
                for input_z in range(0, 16):
                    array_input_y = []
                    for input_y in range(0, 112):
                        array_input_x = []
                        for input_x in range(0, 112):
                            result = 0
                            array_z = []
                            for z in range(3):
                                array_y = []
                                for y in range(3):
                                    array_x = []
                                    for x in range(3):
                                        result = result + input[channel,
                                                                input_z + z, input_y + y, input_x + x] * weight[filter, channel, z, y, x]
                                        print(f'filter index: {filter},filter channel : {channel}\nInput z-axis : {input_z} , Input y-axis : {
                                            input_y}, Input x-axis : {input_x}\nPadding z-axis : {z},Padding y-axis : {y},Padding x-axis : {x}')
                                        array_x = np.append(array_x, result)
                                    array_y = np.append(array_y, array_x)
                                array_z = np.append(array_z, array_y)
                            array_input_x = np.append(array_input_x, array_z)
                        array_input_y = np.append(array_input_y, array_input_x)
                    array_input_z = np.append(array_input_z, array_input_y)
                array_filter_channels = np.append(
                    array_filter_channels, array_input_z)
            output = np.append(output, array_filter_channels)
        np.save('output', output)
        print(output)
        # output = np.append(output, result)
    # print(output)


# def Depthwise(weight, bias, kernal):
#     return weight.size()
#     pass


# def pointwise(weight, bias):
#     pass


# def conv3d(input, weight, bias, stride=1, padding=0):
#     output = f.conv3d(input, input_weight, bias)
#     return output


#   torch.Size([64, 3, 3, 3, 3])
# #   ([input channel,out_channels,kernel_depth,kernel_height,kernel_width])
# Parm_file = "../C3D-main_DSC/models/c3d-pretrained.pth"
# model = torch.load(Parm_file, weights_only=False)
# # bias = 0
# # feature = 0
# output = 0
# first_input = 0
# get_weight = 0
# get_bias = 0
# for name, param in model.items():
#     print(name)
#     print(param.size())
# for name, param in model.items():
#     if (name.find("weight") != -1):
#         feature = param
#         get_weight = 1
#     if (name.find("bias") != -1):
#         bias = param
#         get_bias = 1
#     if (name.find("bias") == -1 and name.find("weight") == -1):
#         break
#     if (get_weight == 1 and get_bias == 1):
#         if (first_input != 0):
#             print(param.size())
#             print(name)
#             output = conv3d(output, feature, bias)
#             print(output.size())
#         else:
#             print(param.size())
#             print(name)
#             first_input = 1
#             output = conv3d(input_data, feature, bias)
#             print(output.size())
#         get_weight = 0
#         get_bias = 0

if __name__ == "__main__":
    # conv3D(input_data, input_weight, input_data)
    # data = Depthwise(input_data, DWC_filter, 1)
    input_data = torch.randn(3, 16, 112, 112)
    input_weight = torch.randn(64, 3, 3, 3, 3)
    DWC_filter = torch.randn(3, 3, 3, 3)
    print("Running...")
    output = Depthwise(input_data, DWC_filter, 1)
    output = f.max_pool3d(output, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    print(output.size())
