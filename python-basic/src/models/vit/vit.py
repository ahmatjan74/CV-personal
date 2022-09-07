import imp
from time import sleep
import torch
from torch import nn, einsum
import torch.nn.functional as F
# import einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# i_tensor = torch.randn(16, 3, 224, 224)
# print(i_tensor.shape)
# o_tensor = rearrange(i_tensor, 'n c h w -> n h c w')
# print(o_tensor.shape)

# i_tensor = torch.randn(16, 3, 224, 224)
# o_tensor = rearrange(i_tensor, 'n c h w -> n c (h w)')
# print(o_tensor.shape)  
# o_tensor = rearrange(i_tensor, 'n c (m1 p1) (m2 p2) -> n c m1 p1 m2 p2', p1=16, p2=14) # 16 3 224 224 -> 16 3 (14 16) (16 14) -> 16 3 14 16 16 14
# print(o_tensor.shape)  

# 判断是不是元祖
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# dim 是纬度， fn是处理函数（attention或者mlp）. -> Norm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # 正则化
        self.norm = nn.LayerNorm(dim)
         # 具体的操作
        self.fn = fn
    def forward(self, x, **kwargs):
        # attention(layerNorm(x)) | mlp(layerNorm(x))
        return self.fn(self.norm(x), **kwargs)


# FeedForward
# FeedForward层由全连接层，配合激活函数GELU和Dropout实现. -> MLP
# 参数dim和hidden_dim分别是输入,输出的维度和中间层的维度，dropout则是dropout操作的概率参数p
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # 前向传播
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
 
 
 
# Multi-Head Attention      
# 参数heads是多头自注意力的头的数目，dim_head是每个头的维度
class Attention(nn.Module):              
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        # 计算最终进行全连接操作时输入神经元的个数
        inner_dim = dim_head * heads 
        # 多头注意力并且输入和输出维度相同时为True
        project_out = not (heads == 1 and dim_head == dim)

        # 多头注意力中“头”的个数
        self.heads = heads
        
        # 缩放操作，论文 Attention is all you need 中有介绍
        self.scale = dim_head ** -0.5

        # 初始化一个Softmax操作
        self.attend = nn.Softmax(dim=-1)
        
        # 对Q、K、V三组向量先进性线性操作
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 获得输入x的维度和多头注意力的“头”数
        b, n, _, h = *x.shape, self.heads
        
        # 先对Q、K、V进行线性操作，然后chunk乘三三份
        # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1) 
        
        # 整理维度，获得Q、K、V
        # q, k, v   (b, h, n, dim_head(64))       
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          

        # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 做Softmax运算
        attn = self.attend(dots)

        # Softmax运算结果与Value向量相乘，得到最终结果
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # 重新整理维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 做线性的全连接操作或者空操作（空操作直接输出out）
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        # Transformer包含多个编码器的叠加
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 编码器包含两大块：自注意力模块和前向传播模块
            # 6layers layer -> (norm(attention), norm(mlp))
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers: # 6 layers
            # 自注意力模块和前向传播模块都使用了残差的模式
            x = attn(x) + x  # residual 
            x = ff(x) + x
        return x

#  image_size = 256,
#         patch_size = 32,
#         num_classes = 1000,
#         dim = 1024,
#         depth = 6,
#         heads = 16,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1
class ViT(nn.Module):
    def __init__(self, 
                 *, 
                 image_size, # 256
                 patch_size, # 32
                 num_classes, #1000
                 dim, #1024
                 depth, # 6   transformer遍历的次数？
                 heads, # 16
                 mlp_dim, #2048
                 pool='cls', 
                 channels=3, 
                 dim_head=64, 
                 dropout=0., 
                 emb_dropout=0.
                ):
        super().__init__()
        image_height, image_width = pair(image_size) # (245, 256)
        patch_height, patch_width = pair(patch_size) # (16, 16)

        # if image_height % patch_height == 0 and image_width % patch_width == 0
        # 执行下面的语句
        assert  image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (256 // 16) * (256 // 16) = 16 * 16 = 256
        # 获取图像切块的个数
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        # patch_dim = 3 * 16 * 16
        # 线性变换时的输入大小，即每一个图像宽、高、通道的乘积, 每一个小块的所有像素个数
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            # 将批量为b,通道为c,高为h*p1,宽为w*p2的图像,转化为批量为b,个数为h*w,维度为p1*p2*c的图像块
            # 即，把b张c通道的图像分割成b*（h*w）张大小为P1*p2*c的图像块
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
            nn.Linear(patch_dim, dim)
        )

        # nn.Parameter()定义可学习参数
        # 位置编码，获取一组正态分布的数据用于训练
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 分类令牌，可训练
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer模块
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        
        # 通过阅读源码可以看到，identity模块不改变输入。直接return input
        # 占位操作
        self.to_latent = nn.Identity() 

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), # 正则化
            nn.Linear(dim, num_classes) # 线性输出
        )

    def forward(self, img):
        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        x = self.to_patch_embedding(img)  
              
       
        # shape (b, n, 1024)
        # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
        b, n, _ = x.shape           

        # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # repeat b次
        
        # 将分类令牌拼接到输入中，x的shape (b, n+1, 1024)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        # 进行位置编码，shape (b, n+1, 1024)           
        x += self.pos_embedding[:, :(n+1)]                 
        x = self.dropout(x)

        # transformer操作
        x = self.transformer(x)                                                 

        # 获取所有向量的平均 或 只需要第一个向量        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]         

        # Identity (b, dim)
        x = self.to_latent(x)                                                   
        print(x.shape)

         #  (b, num_classes)
         # 线性输出
        return self.mlp_head(x)                                                

 
    
    
model_vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024, # 每个向量的维度
        depth = 6,# 用了几次这个Transformer Encoder
        heads = 16,# 多头注意力机制的 多头
        mlp_dim = 2048, # mlp的维度
        dropout = 0.1, # 防止过拟合用的
        emb_dropout = 0.1
    )

img = torch.randn(16, 3, 256, 256) # batch_size: 16, channel: 3, h: 256, w: 256 
preds = model_vit(img) 

print(preds.shape)  # (16, 1000)



        
