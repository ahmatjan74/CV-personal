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



# dim 是纬度， fn是处理函数. -> Norm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# FeedForward
# FeedForward层由全连接层，配合激活函数GELU和Dropout实现. -> MLP
# 参数dim和hidden_dim分别是输入,输出的维度和中间层的维度，dropout则是dropout操作的概率参数p
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
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
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
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
                 depth, # 6
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
        assert  image_height % patch_height == 0 and image_width % patch_width == 0

        # num_patches = (256 // 16) * (256 // 16) = 16 * 16 = 256
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        # patch_dim = 3 * 16 * 16
        patch_dim = channels * patch_height * patch_width
        
        assert pool in {'cls', 'mean'}

        # to_patch_embedding: 
        # einops：灵活和强大的张量操作，可读性强和可靠性好的代码。支持numpy、pytorch、tensorflow等。
        # 代码中Rearrage的意思是将传入的image（3，256，256），
        # 按照（3，（h,p1）,(w,p2))也就是256=h*p1,256 = w*p2，接着把shape变成b (h w) (p1 p2 c)格式的，
        # 这样把图片分成了每个patch并且将patch拉长，方便下一步的全连接层
        # 还有一种方法是采用窗口为16*16，stride 16的卷积核提取每个patch，然后再flatten送入全连接层。

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        # nn.Parameter()定义可学习参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        # 通过阅读源码可以看到，identity模块不改变输入。直接return input
        self.to_latent = nn.Identity() 

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        x = self.to_patch_embedding(img)  
              
        # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
        b, n, _ = x.shape           

        # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  
         # 将cls_token拼接到patch token中去  (b, 65, dim)
        x = torch.cat((cls_tokens, x), dim=1) 
         # 加位置嵌入（直接加）      (b, 65, dim)             
        x += self.pos_embedding[:, :(n+1)]                 
        x = self.dropout(x)

        # (b, 65, dim)
        x = self.transformer(x)                                                 

        # (b, dim)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   

        # Identity (b, dim)
        x = self.to_latent(x)                                                   
        print(x.shape)

         #  (b, num_classes)
        return self.mlp_head(x)                                                

 
    
    
model_vit = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

img = torch.randn(16, 3, 256, 256)
print(img)
preds = model_vit(img) 

print(preds.shape)  # (16, 1000)



        
