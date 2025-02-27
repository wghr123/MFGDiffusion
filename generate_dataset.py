import os.path

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from Module.model import MaskFeatureDiffusion
from safetensors.torch import load_file

pretrained_model = "/home/qixinggroup/wgh123/Model/SD-2-Inpainting/"
safetensors_file = "/home/qixinggroup/wgh123/Project/ModelTest/Use_mask_loss_model/mask_loss0/checkpoint-3000/model.safetensors"
state_dict = load_file(safetensors_file)

# 加载预训练模型和组件
vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")

scheduler = PNDMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

mf_model = MaskFeatureDiffusion(pretrained_model)
mf_model.load_state_dict(state_dict)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移到设备上
vae.to(device)
text_encoder.to(device)

mf_model.to(device)
# torch.manual_seed(1200)
# torch.manual_seed(1200)

# image_path = "D:/Paper Need/Reference-Code/ModelTest/source.png"
# mask_path = "D:/Paper Need/Reference-Code/ModelTest/mask.png"

#image_path = "/home/qixinggroup/wgh123/Project/ModelTest/image.png"
#mask_path = "/home/qixinggroup/wgh123/Project/ModelTest/mask.png"
image_folder_path = "/home/qixinggroup/wgh123/Dataset/SmokeData/testdata/image/"
mask_folder_path = "/home/qixinggroup/wgh123/Dataset/SmokeData/testdata/mask/"

vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
mask_processor = VaeImageProcessor(
    vae_scale_factor=vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
)

crops_coords = None
resize_mode = "default"


def preprocess_image(image_path, mask_path, target_size):
    init_image = Image.open(image_path).convert("RGB").resize(target_size)
    mask = Image.open(mask_path).convert("L").resize(target_size)

    init_image = image_processor.preprocess(
        init_image, height=target_size[0], width=target_size[1], crops_coords=crops_coords, resize_mode=resize_mode
    )
    image = init_image

    mask_condition = mask_processor.preprocess(
        mask, height=target_size[0], width=target_size[1], resize_mode=resize_mode, crops_coords=crops_coords
    )

    return mask_condition, image


# 定义推理函数
def generate_image(prompt, num_inference_steps=50, guidance_scale=7.5, image_path=None, mask_path=None, eta=0.0):
    # 初始化潜在变量
    weight_dtype = torch.float32
    batch_size = 1
    height = 1024
    width = 1024
    mask, image = preprocess_image(image_path, mask_path, target_size=(height, width))
    masked_image = image * (mask < 0.5)
    noise = randn_tensor((batch_size, 4, height // 8, width // 8), device=device)
    latents = noise * scheduler.init_noise_sigma

    # 设置调度器
    scheduler.set_timesteps(num_inference_steps, device=device)

    # 编码提示
    input_ids = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    encoder_hidden_states = text_encoder(input_ids)[0]

    # 获取序列长度
    seq_length = input_ids.shape[-1]

    # 无条件输入
    uncond_input_ids = tokenizer([""] * batch_size, padding="max_length", max_length=seq_length,
                                 return_tensors="pt").input_ids.to(device)
    uncond_encoder_hidden_states = text_encoder(uncond_input_ids)[0]
    encoder_hidden_states_input = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states])

    masked_image_latents = vae.encode(
        masked_image.to(device=device)
    ).latent_dist.sample()
    masked_image_latents = masked_image_latents * vae.config.scaling_factor

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    mask = torch.nn.functional.interpolate(
        mask,
        size=(
            mask.shape[2] // vae_scale_factor,
            mask.shape[3] // vae_scale_factor,
        ),
    )
    mask = mask.to(device=device)
    masked_image = masked_image.to(device)
    masked_image = torch.cat([masked_image] * 2)

    masked_image_latents = torch.cat([masked_image_latents] * 2)
    mask = torch.cat([mask] * 2)

    for i, t in enumerate(scheduler.timesteps):
        #print(i, t)
        # 准备模型输入
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        latent_model_input = torch.cat(
            [latent_model_input, mask, masked_image_latents], dim=1
        )
        # print(latent_model_input.shape, t, encoder_hidden_states_input.shape)

        # 双重前向传播
        with torch.no_grad():
            noise_pred = mf_model(latent_model_input, t, encoder_hidden_states_input, masked_image, mask)

        # 应用指导比例
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 更新潜在变量
        latents = scheduler.step(noise_pred, t, latents)[0]

    # 解码潜在变量为图像
    with torch.no_grad():
        image_latents = vae.decode(latents / vae.config.scaling_factor).sample
        image = (image_latents / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

    return image


print(f"OK")

prompt = ("smoke plume in the mountain and forest")
for i in range(1, 1001):
    image_path = os.path.join(image_folder_path, f"{i}.png")
    mask_path = os.path.join(mask_folder_path, f"{i}.png")
    image = generate_image(prompt, image_path=image_path, mask_path=mask_path)
    #print(image.shape)  # 输出图像的形状

    # 将 numpy 数组转换为图像并保存
    image_pil = Image.fromarray((image[0] * 255).astype(np.uint8))
    image_pil.save(f"/home/qixinggroup/wgh123/Project/ModelTest/Testset/{i}.png")







