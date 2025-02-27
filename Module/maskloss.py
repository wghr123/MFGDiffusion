import torch
import torch.nn.functional as F
import random
from PIL import Image
import numpy as np




def dilate_or_erode_fixed_random(mask_tensor, kernel_size=5, max_iterations=3):
    """
    对给定的二值掩码张量批次执行固定参数的随机膨胀或腐蚀操作，并确保输出尺寸与输入一致。

    :param mask_tensor: 输入的二值掩码张量批次，形状为 (batch_size, 1, H, W)。
    :param kernel_size: 内核大小，默认为5。
    :param max_iterations: 最大迭代次数，默认为3。
    :return: 经过随机膨胀或腐蚀后的掩码张量批次，尺寸与输入一致。
    """
    batch_size, channels, height, width = mask_tensor.shape
    # 确保mask是二值的
    mask_tensor = (mask_tensor > 0.5).float()

    output_masks = []
    for i in range(batch_size):
        sample_mask = mask_tensor[i].unsqueeze(0)  # 添加批次维度

        iterations = random.randint(1, max_iterations)
        for _ in range(iterations):
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask_tensor.device)

            if random.random() < 0.5:
                # 随机选择膨胀
                dilated_mask = F.conv2d(sample_mask, kernel, padding=kernel_size // 2, stride=1)
                sample_mask = (dilated_mask > 0).float()
            else:
                # 随机选择腐蚀
                eroded_mask = -F.max_pool2d(-sample_mask, kernel_size, stride=1, padding=kernel_size // 2,
                                            ceil_mode=True)
                sample_mask = eroded_mask

        output_masks.append(sample_mask.squeeze(0))  # 去除批次维度

    return torch.stack(output_masks)  # 将所有处理过的样本堆叠回一个批次张量
