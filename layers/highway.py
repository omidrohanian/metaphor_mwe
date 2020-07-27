import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):

    def __init__(self, input_size, bias=-2.0):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(bias)

    def forward(self, input):
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = F.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


