import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import sparse_mx_to_torch_sparse_tensor

class GC_withres(Module):
    """
    res conv
    """
    def __init__(self, in_features, out_features,smooth,bias=True):
        super(GC_withres, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.smooth = smooth
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        stdv = 1. / math.sqrt(self.weight.size(1))
#        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        # adj is extracted from the graph structure
        support = torch.mm(input, self.weight)
        I_n = sp.eye(adj.shape[0])
        I_n = sparse_mx_to_torch_sparse_tensor(I_n).cuda()

        degrees = torch.sparse.sum(adj,0)
        D = degrees
        D = D.to_dense() # transfer D from sparse tensor to normal torch tensor
        D = torch.pow(D, -1)
        D = D.unsqueeze(dim=1)
        D_inv_x = D*support
        W_D_inv_x = torch.spmm(adj,D_inv_x)

        sctf = 0.5*support + 0.5*W_D_inv_x
        output = support + self.smooth*sctf
        output = output/(1+self.smooth) 
#        output = torch.spmm((I_n+self.smooth*adj)/(1+self.smooth), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


