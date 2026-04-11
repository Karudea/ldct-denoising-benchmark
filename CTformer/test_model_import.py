from CTformer import CTformer
import torch

x = torch.randn(1, 1, 64, 64)
model = CTformer(
    img_size=64,
    tokens_type='performer',
    embed_dim=64,
    depth=1,
    num_heads=8,
    kernel=4,
    stride=4,
    mlp_ratio=2.0,
    token_dim=64
)
y = model(x)
print(y.shape)