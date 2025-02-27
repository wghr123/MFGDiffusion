import torch
from torchvision import models, transforms
from PIL import Image
from diffusers import UNet2DConditionModel
from transformers import CLIPProcessor, CLIPModel


from Module.attention import BaseCrossAttention
from Module.attention import CrossAttentionFusion



class ExtractorModel(torch.nn.Module):
    def __init__(self):
        super(ExtractorModel, self).__init__()
        # get resnet model
        model = models.resnet152()
        weight = torch.load("/home/qixinggroup/wgh123/Model/Image Encoder/resnet152-b121ed2d.pth")
        model.load_state_dict(weight)

        self.resnet = model

        self.feature = None
        self._attach_layer_hook()


    def get_feature(self, module, inputs, outputs):
        x = outputs
        self.feature = x
        #print(x.shape)

    def _attach_layer_hook(self):
        for n, m in self.resnet.named_modules():
            # Last layer in downsampling block
            # if resnet, n=="avgpool"
            if n == "layer4.2.conv3":
                m.register_forward_hook(self.get_feature)


    def forward(self, images):
        output = self.resnet(images)
        return self.feature



'''
#测试图像特征提取模块

transform = transforms.Compose([
    transforms.Resize(256),  # 缩放到 256x256
    transforms.CenterCrop(224),  # 中心裁剪到 224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 读取图片
image_path = 'D:/Paper Need/Reference-Code/ModelTest/Data/2023-04-03---23--11--34.403961--00040.jpg'
image = Image.open(image_path).convert('RGB')

# 应用预处理变换
image_tensor = transform(image)

# 添加批量维度
image_tensor = image_tensor.unsqueeze(0)  # 形状变为 (1, 3, 224, 224)
print("Image tensor shape:", image_tensor.shape)


res = ExtractorModel()
with torch.no_grad():  # 关闭梯度计算
    output = res(image_tensor)
print("图像特征提取模块返回的结果：")
print(output.shape)


'''



class MaskFeatureDiffusion(torch.nn.Module):

    def __init__(self, model_name):
        super(MaskFeatureDiffusion, self).__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.extractor = ExtractorModel()
        self.extractor.eval()

        # init cross-attention layer
        self.attn_down1 = BaseCrossAttention(embed_dim=320, num_heads=8, hidden_dim=256)
        self.attn_down2 = CrossAttentionFusion(embed_dim=640, num_heads=8, hidden_dim=256)
        self.attn_mid = BaseCrossAttention(embed_dim=1280, num_heads=8, hidden_dim=256)
        self.attn_up2 = CrossAttentionFusion(embed_dim=640, num_heads=8, hidden_dim=256)
        self.attn_up1 = BaseCrossAttention(embed_dim=320, num_heads=8, hidden_dim=256)

        # init
        self.linear_transform1 = torch.nn.Linear(2048, 320)
        self.linear_transform2 = torch.nn.Linear(2048, 640)
        self.linear_transform3 = torch.nn.Linear(2048, 1280)
        self.linear_transform4 = torch.nn.Linear(2048, 640)
        self.linear_transform5 = torch.nn.Linear(2048, 320)

        # Attach Pytorch hooks that get intermediate feature fusion
        self._attach_layer_hook()

    def get_features(self, masked_image, mask):
        with torch.no_grad():
            self.feature1 = self.extractor(masked_image)
            # 这里mask要转成三通道
            mask = mask.repeat(1, 3, 1, 1)
            self.feature2 = self.extractor(mask)

    # 这里定义特征融合函数，后面传入hook注册函数里
    def mask_feature_fusion(self, module, inputs, outputs):
        # 这里要注意，outputs直接就是tensor了，inputs不一样，是tuple，所以需要先提取inputs[0]
        out = outputs
        feat1 = self.feature1
        feat2 = self.feature2
        # print(out.shape, feat1.shape, feat2.shape)

        if out.shape[0] != feat1.shape[0]:
            print("出错了")
        # 提取特征维度, 这里h和w好像写反了，不过关系应该不大，因为后面也是反着写的
        b1, e1, w1, h1 = out.shape[0], out.shape[1], out.shape[2], out.shape[3]
        b1, e2, w2, h2 = feat1.shape[0], feat1.shape[1], feat1.shape[2], feat1.shape[3]

        # 调整特征维度
        out_flat = out.view(b1, e1, -1).permute(0, 2, 1)
        feat1_flat = feat1.view(b1, e2, -1).permute(0, 2, 1)
        feat2_flat = feat2.view(b1, e2, -1).permute(0, 2, 1)


        # 要根据不同的层来确定如何融合和投影，这个可以使用if来实现
        # 下采样fine-layer,这一层融合masked_image,选择down_blocks.0.resnets.1.conv2
        if module is self.unet.down_blocks[0].resnets[1].conv2:
            feat1_trans = self.linear_transform1(feat1_flat)
            q = out_flat
            k1 = feat1_trans
            v1 = feat1_trans

            fused_output = self.attn_down1(q, k1, v1)
            fused_output = fused_output.view(b1, e1, w1, h1)
            return fused_output

        # 下采样 fine-coarse-layer，这一层融合mask和masked_image，选择down_blocks.1.resnets.1.conv2
        if module is self.unet.down_blocks[1].resnets[1].conv2:
            feat1_trans = self.linear_transform2(feat1_flat)
            feat2_trans = self.linear_transform2(feat2_flat)
            q = out_flat
            k1 = feat1_trans
            v1 = feat1_trans
            k2 = feat2_trans
            v2 = feat2_trans

            fused_output = self.attn_down2(q, k1, k2, v1, v2)
            fused_output = fused_output.view(b1, e1, w1, h1)
            return fused_output

        # 中间层 coarse-layer，这一层融合mask，选择mid_block.resnets.1.conv2
        if module is self.unet.mid_block.resnets[1].conv2:
            feat2_trans = self.linear_transform3(feat2_flat)
            q = out_flat
            k1 = feat2_trans
            v1 = feat2_trans

            fused_output = self.attn_mid(q, k1, v1)
            fused_output = fused_output.view(b1, e1, w1, h1)
            return fused_output

        # 上采样 fine-coarse-layer，这一层融合mask和masked_image，选择up_blocks.2.resnets.2.conv2
        if module is self.unet.up_blocks[2].resnets[2].conv2:
            feat1_trans = self.linear_transform4(feat1_flat)
            feat2_trans = self.linear_transform4(feat2_flat)
            q = out_flat
            k1 = feat1_trans
            v1 = feat1_trans
            k2 = feat2_trans
            v2 = feat2_trans

            fused_output = self.attn_up2(q, k1, k2, v1, v2)
            fused_output = fused_output.view(b1, e1, w1, h1)
            return fused_output

        # 上采样fine-layer,这一层融合masked_image，选择up_blocks.3.resnets.2.conv2
        if module is self.unet.up_blocks[3].resnets[2].conv2:
            feat1_trans = self.linear_transform5(feat1_flat)
            q = out_flat
            k1 = feat1_trans
            v1 = feat1_trans

            fused_output = self.attn_up1(q, k1, v1)
            fused_output = fused_output.view(b1, e1, w1, h1)
            return fused_output

    def _attach_layer_hook(self):
        for n, m in self.unet.named_modules():
            # 根据上面选定的层注册hook
            if n == "down_blocks.0.resnets.1.conv2":
                m.register_forward_hook(self.mask_feature_fusion)
            if n == "down_blocks.1.resnets.1.conv2":
                m.register_forward_hook(self.mask_feature_fusion)
            if n == "mid_block.resnets.1.conv2":
                m.register_forward_hook(self.mask_feature_fusion)
            if n == "up_blocks.2.resnets.2.conv2":
                m.register_forward_hook(self.mask_feature_fusion)
            if n == "up_blocks.3.resnets.2.conv2":
                m.register_forward_hook(self.mask_feature_fusion)


    def forward(self, latent, timestep, encoder_hidden_states, masked_image, mask):

        self.get_features(masked_image, mask)
        return self.unet(latent, timestep, encoder_hidden_states).sample



