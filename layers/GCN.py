import torch, math
import torch.nn as nn
import copy

from torch.nn.parameter import Parameter
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn


class GraphConvolution(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 bias = True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj.float().cuda(), support)

        if self.bias is not None:
            return output + self.bias
        return output


class ABGCN(nn.Module):
    """Attention-based Graph Convolutional Network"""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_heads,
                 num_layers,
                 d_model = 768,
                 alpha = 0.3,
                 beta = 0.5,
                 dropout = 0.2):
        super(ABGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.beta = beta
        self.heads = num_heads

        self.attn = MultiHeadAttention(h=self.heads, d_model=d_model)

        self.gc = nn.ModuleList([GraphConvolution(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Linear(self.heads*d_model, d_model)

    def forward(self, inputs, A, heads):
        A = A.float().to(device) + torch.transpose(A, 2, 1).float().to(device) + torch.eye(A.shape[1]).repeat(A.shape[0],1,1).float().to(device)
        A = A.to(device)
        
        input_att = self.attn(inputs, inputs)             

        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(input_att, 1, dim=1)]

        multi_head_list = []
        for h in range(self.heads):
            A_final = self.alpha * attn_adj_list[h].to(device) + (1-self.alpha)* A.to(device)

            for i in range(self.num_layers):
                inputs = inputs * self.beta + (1 - self.beta) * F.relu(self.gc[i](inputs, A_final))
                inputs = self.dropout(inputs)
            multi_head_list.append(F.relu(inputs))

        final_output = torch.cat(multi_head_list, dim=2)
        final_output = self.linear(self.dropout2(final_output))

        return final_output
