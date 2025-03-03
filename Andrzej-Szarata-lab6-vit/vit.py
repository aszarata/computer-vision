import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.linear1 = nn.Linear(embed_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.multihead_attn(y, y, y)
        y = self.dropout(y)
        x = x + y

        y = self.norm2(x)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        x = x + y

        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.linear = nn.Linear(patch_size * patch_size * in_channels, embed_size)

    def forward(self, x):
        num_patches_h = self.img_size // self.patch_size
        num_patches_w = self.img_size // self.patch_size
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.shape[0], num_patches_h, self.patch_size, num_patches_w, self.patch_size, x.shape[3])
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(x.shape[0], -1, self.patch_size * self.patch_size * x.shape[-1])
        x = self.linear(x)
        return x
    

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size, num_heads, depth, n_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_size)

        self.positional_encoding = nn.Parameter(torch.randn(1, self.patch_embedding.n_patches, embed_size))

        transformers = [
            TransformerEncoderLayer(
                embed_size=embed_size, 
                num_heads=num_heads, 
                hidden_dim=512
            )
        ] * depth
        
        self.transformers = nn.Sequential(*transformers)

        self.linear = nn.Linear(embed_size, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        x += self.positional_encoding
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x
