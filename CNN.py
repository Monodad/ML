import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as f
# Trying to use for loop to implement 3D convolution.
# 16x112x112 image with 3 channel(R-G-B)
input_data = torch.randn(3, 16, 112, 112)
input_weight = torch.randn(64, 3, 3, 3, 3)
# input_data = torch.randn(3, 16)
# def cal_conv3D(input:Tensor,feature:Tensor,)


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
#   ([input channel,out_channels,kernel_depth,kernel_height,kernel_width])
# Parm_file = "../C3D-main_DSC/models/c3d-pretrained.pth"
# model = torch.load(Parm_file, weights_only=False)
# bias = 0
# feature = 0
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
    conv3D(input_data, input_weight, input_data)
