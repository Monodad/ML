import torch
import numpy as np


def save_array_to_txt(array, filename):
    if torch.is_tensor(array):
        array = array.numpy()

    if array.size == 0:
        print(f"警告: 儲存 {filename} 時遇到空陣列")
        return

    with open(filename, 'w') as f:
        array_str = np.array2string(
            array,
            separator=', ',    # 用逗號和空格分隔
            precision=8,       # 8位小數
            suppress_small=True,  # 抑制科學記號
            threshold=np.inf,   # 顯示所有元素
            max_line_width=np.inf  # 不換行
        )
        f.write(array_str)
