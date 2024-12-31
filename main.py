from CNN import *
from Quantized import *
import os
from save_to_txt import *


class Item:
    def __init__(self, name, param):
        self.name = name
        self.param = param

    def __repr__(self):
        return f"Item(name={self.name},param={self.param})"


def folder_create(name: str):
    try:
        os.makedirs(f"./{name}")
        print(f"Nested directories '{name}' created successfully.")
    except FileExistsError:
        print(f"One or more directories in '{name}' already exist.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


# MP3d DWC PWC QTZ
data = [
    Item(name="input", param={"size": (3, 16, 112, 112)}),
    Item(name="DWC", param={"filter": (3, 3, 3, 3)}),
    Item(name="MP3d", param={"kernal_size": (1, 2, 2), "stride": (1, 2, 2)}),
    Item(name="PWC", param={"filter": (64, 3, 1, 1, 1)}),
    Item(name="DWC", param={"filter": (64, 3, 3, 3)}),
    Item(name="MP3d", param={"kernal_size": (2, 2, 2), "stride": (2, 2, 2)}),
    Item(name="PWC", param={"filter": (128, 64, 1, 1, 1)}),
    Item(name="DWC", param={"filter": (128, 3, 3, 3)}),
    Item(name="PWC", param={"filter": (256, 128, 1, 1, 1)}),
    Item(name="DWC", param={"filter": (256, 3, 3, 3)}),
    Item(name="MP3d", param={"kernal_size": (2, 2, 2), "stride": (2, 2, 2)}),
    Item(name="PWC", param={"filter": (256, 256, 1, 1, 1)}),
    Item(name="DWC", param={"filter": (256, 3, 3, 3)}),
    Item(name="PWC", param={"filter": (512, 256, 1, 1, 1)}),
    Item(name="DWC", param={"filter": (512, 3, 3, 3)}),
    Item(name="MP3d", param={"kernal_size": (2, 2, 2), "stride": (2, 2, 2)}),
    Item(name="PWC", param={"filter": (512, 512, 1, 1, 1)}),
    Item(name="DWC", param={"filter": (512, 3, 3, 3)}),
    Item(name="PWC", param={"filter": (512, 512, 1, 1, 1)}),
    Item(name="DWC", param={"filter": (512, 3, 3, 3)}),
    Item(name="MP3d", param={"kernal_size": (2, 2, 2),
         "stride": (2, 2, 2), "padding": (0, 1, 1)}),
    Item(name="PWC", param={"filter": (512, 512, 1, 1, 1)})
]
Layer = ["DWC", "MP3d", "PWC", "DWC", "MP3d", "@PWC", "@DWC", "@PWC", "@DWC", "@MP3d",
         "@PWC", "@DWC", "@PWC", "@DWC", "#P3d", "@PWC", "@DWC", "@PWC", "DWC", "MP3d", "PWC"]
input_data = torch.randn(3, 16, 112, 112)

Next_layer_input = 0


def savefile(path: str, input: Tensor):
    save_array_to_txt(input, path)
    np.save(path, input.numpy())


def loadfile(path: str, QUAN: bool):
    if QUAN:
        return torch.from_numpy(np.load(f"{path}_int.npy"))

    else:
        return torch.from_numpy(np.load(f"{path}.npy"))


if __name__ == "__main__":
    layer_num = 0
    for item in data:
        filename = f"{item.name}_{layer_num}"
        print(filename)
        match item.name:
            case "DWC":
                folder_create(filename)
                savefile(f"./{filename}/{filename}_input", Next_layer_input)
                filter = torch.randn(item.param.get("filter"))
                savefile(f"./{filename}/{filename}_filter", filter)
                filter = quantized_int8(filter.float())
                savefile(f"./{filename}/{filename}_filter_int8", filter)
                Next_layer_input = Depthwise(Next_layer_input, filter, 1)
                savefile(f"./{filename}/{filename}_out", Next_layer_input)
                print(Next_layer_input.size())
                pass
            case "PWC":
                folder_create(filename)
                savefile(f"./{filename}/{filename}_input", Next_layer_input)
                filter = torch.randn(item.param.get("filter"))
                savefile(f"./{filename}/{filename}_filter", filter)
                filter = quantized_int8(filter.float())
                savefile(f"./{filename}/{filename}_filter_int8", filter)
                Next_layer_input = conv3D(
                    Next_layer_input, filter, padding=False)
                Next_layer_input = accumulation(Next_layer_input, 1)
                savefile(f"./{filename}/{filename}_out", Next_layer_input)
                Next_layer_input = quantized_int8(Next_layer_input.float())
                savefile(f"./{filename}/{filename}_out_int", Next_layer_input)
                print(Next_layer_input.size())
                pass
            case "MP3d":
                folder_create(filename)
                savefile(f"./{filename}/{filename}_input", Next_layer_input)
                Next_layer_input = f.max_pool3d(
                    Next_layer_input, kernel_size=item.param.get("kernal_size"), stride=item.param.get("stride"))
                savefile(f"./{filename}/{filename}_out", Next_layer_input)

            case "input":
                try:

                    Next_layer_input = torch.from_numpy(np.load('input.npy'))
                except:
                    input = torch.randn(item.param.get("size"))
                    input = quantized_int8(
                        input.float())
                    np.save('input', input.numpy())
                    save_array_to_txt(input, './Hello/input.txt')
                    Next_layer_input = input
            case _:
                pass
        layer_num = layer_num + 1
