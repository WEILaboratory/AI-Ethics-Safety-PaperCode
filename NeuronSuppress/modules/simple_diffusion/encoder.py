import torch
from torch import nn

class TextEmbedding(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.embed = nn.Embedding(49408, 768)
        self.pos_embed = nn.Embedding(77, 768)
 
        self.register_buffer('pos_ids', torch.arange(77).unsqueeze(dim=0))  #我们需要保存一个状态，但是这个状态不能看作成为模型参数。

    def forward(self, input):
        # input -> [b,77]

        #[b, 77] -> [b, 77, 768]
        embed = self.embed(input)

        #[1, 77] -> [1, 77, 768]
        pos_embed = self.pos_embed(self.pos_ids)

        #[b, 77, 768]
        return embed + pos_embed
    
# print(TextEmbedding()(torch.ones(2,77).long()).shape)    

def get_mask(b):
    mask = torch.empty(b, 77, 77)

    #上三角的部分置为负无穷
    mask.fill_(-float('inf'))

    #对角线和以下的位置为0
    mask.triu_(1)
    return mask.unsqueeze(1)

#   MultiHeadedAttention
class Attention(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)
        self.out = nn.Linear(768, 768)

    def forward(self, x):
        #x -> [b, 77, 768]

        batch, high, width = x.shape

        q = self.q(x) * 0.125   # Q * K^T / sqrt(d_k)   d_k=64
        k = self.k(x)
        v = self.v(x)

        #拆分注意力头
        #[b, 77, 768] -> [b, 77, 12, 64] -> [b, 12, 77, 64] -> [b*12, 77, 64]
        q, k, v = [
            x.reshape(batch, 77, 12, 64).transpose(1, 2).reshape(batch * 12, 77, 64) 
            for x in (q, k, v)
        ]

        attention = torch.bmm(q, k.transpose(1, 2))  # Q * K^T  [b*12, 77, 64] * [b*12, 64, 77] -> [b*12, 77, 77]
        attention = attention.reshape(batch, 12, 77, 77)

        attention = attention + get_mask(attention.shape[0]).to(attention.device)   #[b, 12, 77, 77] + [b, 1, 77, 77] -> [b, 12, 77, 77]
        attention = attention.reshape(batch * 12, 77, 77)   #[b, 12, 77, 77] -> [b*12, 77, 77]

        attention = attention.softmax(dim=-1)
        attention = torch.bmm(attention, v)

        attention = attention.reshape(batch, 12, 77, 64).transpose(1, 2).reshape(batch, 77, 768)    #[b*12, 77, 64] -> [b, 12, 77, 64] -> [b, 77, 12, 64] -> [b, 77, 768]

        return self.out(attention)

# print(Attention()(torch.randn(2, 77, 768)).shape)        

class CLIPEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.s1 = nn.Sequential(
            nn.LayerNorm(768),
            Attention()
        )

        self.s2 = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 3072),
        )

        self.s3 = nn.Linear(3072, 768)

    def forward(self, x):
        x = x + self.s1(x)
        res = x
        x = self.s2(x)
        x = x * (x * 1.702).sigmoid()
        return res + self.s3(x)

# print(CLIPEncoder()(torch.randn(2, 77, 768)).shape)
