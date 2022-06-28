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

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.mlp = nn.Sequential(nn.Linear(in_features, 64),\
                nn.LeakyReLU(),\
                nn.Dropout(dropout),\
                nn.Linear(32, 32),\
                nn.LeakyReLU(),\
                nn.Dropout(dropout),\
                nn.Linear(16, out_features),\
                nn.LeakyReLU()
                )
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
#        self.W1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
#        self.W2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
#        self.W3 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W3.data, gain=1.414)
#        self.W4 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W4.data, gain=1.414)
#        self.W5 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W5.data, gain=1.414)
#        self.W6 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W6.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
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
        h_sct1,h_sct2,h_sct3 = scattering_diffusion(adj,support0)
        h_sct1 = torch.FloatTensor.abs_(h_sct1)**1
        h_sct2 = torch.FloatTensor.abs_(h_sct2)**1
        h_sct3 = torch.FloatTensor.abs_(h_sct3)**1
#        h_A =  torch.spmm(A_nor, torch.mm(input, self.W1))
#        h_A2 = torch.spmm(A_nor,torch.spmm(A_nor, torch.mm(input, self.W2)))
#        h_A3 = torch.spmm(A_nor,torch.spmm(A_nor,torch.spmm(A_nor, torch.mm(input, self.W3))))
#        h_sct1 = torch.FloatTensor.abs_(torch.spmm(P_sct1, torch.mm(input, self.W4)))**1
#        h_sct2 = torch.FloatTensor.abs_(torch.spmm(P_sct2, torch.mm(input, self.W5)))**1
#        h_sct3 = torch.FloatTensor.abs_(torch.spmm(P_sct3, torch.mm(input, self.W6)))**1

        '''
        h_A: N by out_features
        h_A^2: N by out_features
        h_A^3: ...
        h_sct_1: ...
        h_sct_2: ...
        h_sct_3: ...
        '''
        h = support0


        ####
        ####
#        h = torch.spmm(A_nor, torch.mm(input, self.W))
        a_input_A = torch.cat([h,h_A]).view(N, -1, 2 * self.out_features)
        a_input_A2 = torch.cat([h,h_A2]).view(N, -1, 2 * self.out_features)
        a_input_A3 = torch.cat([h,h_A3]).view(N, -1, 2 * self.out_features)
        a_input_sct1 = torch.cat([h,h_sct1]).view(N, -1, 2 * self.out_features)
        a_input_sct2 = torch.cat([h,h_sct2]).view(N, -1, 2 * self.out_features)
        a_input_sct3 = torch.cat([h,h_sct3]).view(N, -1, 2 * self.out_features)
        a_input =  torch.cat((a_input_A,a_input_A2,a_input_A3,a_input_sct1,a_input_sct2,a_input_sct3),1).view(N, 6, -1) 
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        '''
        a_input shape
        e shape: (N,-1)
        attention shape: (N,out_features)
        '''

        attention = F.softmax(e, dim=1).view(N, 6, -1)
#        print('attention on %d nodes'%N)
#        print(attention[data_index])
        h_all = torch.cat((h_A.unsqueeze(dim=2),h_A2.unsqueeze(dim=2),h_A3.unsqueeze(dim=2),\
                h_sct1.unsqueeze(dim=2),h_sct2.unsqueeze(dim=2),h_sct3.unsqueeze(dim=2)),dim=2).view(N, 6, -1)
        h_prime = torch.mul(attention, h_all) # element eise product
        h_prime = torch.mean(h_prime,1)
        
#        return h_prime
        return [h_prime,attention]
