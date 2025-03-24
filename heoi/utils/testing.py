import torch
import torch.nn as nn

"flatten"
data = torch.randn(2,20,10)
print(data.shape)
data = data.flatten(1)
print(data.shape)

"nn.Embedding.weight"
class WeightLayer(nn.Module):
    def __init__(self, weight_len):
        super(WeightLayer, self).__init__()
        self.init_weight = nn.Embedding(weight_len, 1)

    def forward(self, x):
        weight = self.init_weight.weight.clone()
        print(x.shape)
        print(weight.shape)
    
        return x * weight

# 权重的长度注意和序列长度保持一致    
weight_layer = WeightLayer(20)
# 输入序列[Batch Size，sequence length, Hidden Size]
data = torch.randn(2,20,128)
weight_layer(data)

