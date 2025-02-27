import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义交叉注意力层
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(CrossAttentionFusion, self).__init__()
        self.attention1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q, k1, k2, v1, v2):
        # 第一次交叉注意力
        attn_output1, _ = self.attention1(q.permute(1, 0, 2), k1.permute(1, 0, 2), v1.permute(1, 0, 2))
        # 第二次交叉注意力
        attn_output2, _ = self.attention2(q.permute(1, 0, 2), k2.permute(1, 0, 2), v2.permute(1, 0, 2))

        # 融合结果
        fused_output = torch.cat((attn_output1, attn_output2), dim=-1)

        # 通过多层感知机
        fused_output = self.linear1(fused_output)
        fused_output = F.relu(fused_output)
        fused_output = self.dropout(fused_output)
        fused_output = self.linear2(fused_output)


        # 残差连接
        fused_output = fused_output + q.permute(1, 0, 2)
        fused_output = self.norm(fused_output)

        return fused_output.permute(1, 2, 0)



class BaseCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(BaseCrossAttention, self).__init__()
        self.attention1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        # 第一次交叉注意力
        fused_output, _ = self.attention1(q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2))

        #通过多层感知机
        fused_output = self.linear1(fused_output)
        fused_output = F.relu(fused_output)
        fused_output = self.dropout(fused_output)
        fused_output = self.linear2(fused_output)

        # 残差连接
        fused_output = fused_output + q.permute(1, 0, 2)
        fused_output = self.norm(fused_output)

        return fused_output.permute(1, 2, 0)





'''
#用于测试3个向量的联合交叉注意力机制，这是目前的结果，已经完善

# 假设输入特征向量
b1, e1, w1, h1 = 2, 64, 32, 32
b1, e2, w2, h2 = 2, 128, 16, 16

# 创建随机特征向量
feat1 = torch.randn(b1, e1, w1, h1)
feat2 = torch.randn(b1, e2, w2, h2)
feat3 = torch.randn(b1, e2, w2, h2)

# 调整特征维度
feat1_flat = feat1.view(b1, e1, -1).permute(0, 2, 1)  # 形状变为 [b1, w1*h1, e1]
feat2_flat = feat2.view(b1, e2, -1).permute(0, 2, 1)  # 形状变为 [b1, w2*h2, e2]
feat3_flat = feat3.view(b1, e2, -1).permute(0, 2, 1)  # 形状变为 [b1, w2*h2, e2]

# 定义线性变换层，将 feat2 和 feat3 的特征维度调整为 e1
linear_transform = nn.Linear(e2, e1)

# 应用线性变换
feat2_flat = linear_transform(feat2_flat)
feat3_flat = linear_transform(feat3_flat)


# 初始化交叉注意力层
embed_dim = e1  # 假设嵌入维度为 e1
num_heads = 8  # 头数
hidden_dim = 256  # 隐藏层维度
fusion_layer = CrossAttentionFusion(embed_dim, num_heads, hidden_dim)

# 应用交叉注意力机制
q = feat1_flat
k1 = feat2_flat
k2 = feat3_flat
v1 = feat2_flat
v2 = feat3_flat

fused_output = fusion_layer(q, k1, k2, v1, v2)

# 调整输出维度
fused_output = fused_output.view(b1, e1, w1, h1)  # 形状变为 [b1, e1, w1, h1]

print(fused_output.shape)  # 输出: torch.Size([2, 64, 32, 32])

'''



'''
#这一段是对基础注意力模型的测试，适用于两个向量之间进行特征融合

# 假设输入特征向量
b1, e1, w1, h1 = 2, 64, 32, 32
b1, e2, w2, h2 = 2, 128, 16, 16

# 创建随机特征向量
feat1 = torch.randn(b1, e1, w1, h1)
feat2 = torch.randn(b1, e2, w2, h2)


# 调整特征维度
feat1_flat = feat1.view(b1, e1, -1).permute(0, 2, 1)  # 形状变为 [b1, w1*h1, e1]
feat2_flat = feat2.view(b1, e2, -1).permute(0, 2, 1)  # 形状变为 [b1, w2*h2, e2]


# 定义线性变换层，将 feat2 和 feat3 的特征维度调整为 e1
linear_transform = nn.Linear(e2, e1)

# 应用线性变换
feat2_flat = linear_transform(feat2_flat)



# 初始化交叉注意力层
embed_dim = e1  # 假设嵌入维度为 e1
num_heads = 8  # 头数
hidden_dim = 256  # 隐藏层维度
fusion_layer = BaseCrossAttention(embed_dim, num_heads, hidden_dim)

# 应用交叉注意力机制
q = feat1_flat
k1 = feat2_flat

v1 = feat2_flat


fused_output = fusion_layer(q, k1, v1)

# 调整输出维度
fused_output = fused_output.view(b1, e1, w1, h1)  # 形状变为 [b1, e1, w1, h1]

print(fused_output.shape)  # 输出: torch.Size([2, 64, 32, 32])
'''
