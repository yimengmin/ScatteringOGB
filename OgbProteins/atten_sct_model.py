import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion import GCN_diffusion,scattering_diffusion
class ScattterAttentionLayer(nn.Module):
    '''
    Scattering Attention Layer
    '''
    def __init__(self,in_features, out_features, dropout, alpha=0.1):
        super(ScattterAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
#        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W.data, gain=1.414)
#        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
#        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.mlp = nn.Linear(in_features, out_features,bias=False)
        self.a = nn.Linear(2*out_features, 1,bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self,input,adj,data_index = [1,2,3,4]):
        # input is the feature
#        support0 = torch.mm(input, self.W)
        support0 = self.mlp(input)
        gcn_diffusion_list = GCN_diffusion(adj,3,support0)
        N = support0.size()[0]
        h_A =  gcn_diffusion_list[0]
        h_A2 =  gcn_diffusion_list[1]
        h_A3 =  gcn_diffusion_list[2]
#        h_A = self.leakyrelu(h_A)
#        h_A2 = self.leakyrelu(h_A2)
#        h_A3 = self.leakyrelu(h_A3)
        h_sct1,h_sct2,h_sct3 = scattering_diffusion(adj,support0)
        h_sct1 = torch.FloatTensor.abs_(h_sct1)**1
        h_sct2 = torch.FloatTensor.abs_(h_sct2)**1
        h_sct3 = torch.FloatTensor.abs_(h_sct3)**1
        '''
        h_A: N by out_features
        h_A^2: N by out_features
        h_A^3: ...
        h_sct_1: ...
        h_sct_2: ...
        h_sct_3: ...
        '''
        h = support0
        a_input_A = torch.hstack((h, h_A)).unsqueeze(1)
        a_input_A2 = torch.hstack((h, h_A2)).unsqueeze(1)
        a_input_A3 = torch.hstack((h, h_A3)).unsqueeze(1)
        a_input_sct1 = torch.hstack((h, h_sct1)).unsqueeze(1)
        a_input_sct2 = torch.hstack((h, h_sct2)).unsqueeze(1)
        a_input_sct3 = torch.hstack((h, h_sct3)).unsqueeze(1)
        a_input =  torch.cat((a_input_A,a_input_A2,a_input_A3,a_input_sct1,a_input_sct2,a_input_sct3),1).view(N, 6, -1) 
#        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) #plain gat, we got > 76 accuracy on ogb protein
#        e = torch.matmul(self.leakyrelu(a_input), self.a).squeeze(2) # GATv2
        e = self.leakyrelu(self.a(a_input).squeeze(2))
        e = self.a(self.leakyrelu(a_input).squeeze(2))# GATv2
        '''
        a_input shape
        e shape: (N,-1)
        attention shape: (N,out_features)
        '''
        attention = F.softmax(e, dim=1).view(N, 6, -1)
#        print(attention[data_index])
        h_all = torch.cat((h_A.unsqueeze(dim=2),h_A2.unsqueeze(dim=2),h_A3.unsqueeze(dim=2),\
                h_sct1.unsqueeze(dim=2),h_sct2.unsqueeze(dim=2),h_sct3.unsqueeze(dim=2)),dim=2).view(N, 6, -1)
        h_prime = torch.mul(attention, h_all) # element eise product
        h_prime = torch.mean(h_prime,1)
        return [h_prime,attention]
