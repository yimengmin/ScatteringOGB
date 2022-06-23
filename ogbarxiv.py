#import pickle 
#print(pickle.format_version)
#print(pickle.__version__)
from ogb.nodeproppred import Evaluator
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
import time

import os.path as osp
import scipy.sparse as sp
import argparse
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
import torch.nn as nn
from torch_geometric.datasets import WikiCS
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.transforms as T
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'WikiCS')
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import normalize_adjacency_matrix,normalizemx
from utils import normalize_adjacency_matrix,accuracy,scattering1st,sparse_mx_to_torch_sparse_tensor
from layers import GC_withres
import torch.optim as optim
import numpy as np
from models import SCT_GAT_ogbarxiv

### use gcn
#from torch_geometric.nn import GCNConv, ChebConv  # noqa
from layers import GC_withres,GC
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1', type=float, default=0.0,
                    help='Weight decay (L1 loss on parameters).')
parser.add_argument('--hid', type=int, default=60,
                    help='Number of hidden units.')
parser.add_argument('--nheads', type=int, default=10,
                    help='Number of heads in attention mechism.')
parser.add_argument('--data_spilt', type=int, default=0)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='Smooth for Res layer')
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--nlayers', type=int, default=2,
                    help='num of layers')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    print('Traning on Cuda,yeah!')
    torch.cuda.manual_seed(args.seed)


#from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import MultiStepLR,StepLR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Num of feat:1639
dataset = PygNodePropPredDataset(name="ogbn-arxiv")
split_idx = dataset.get_idx_split()


train_idx = split_idx['train'].to(device)
print(train_idx)
data = dataset[0]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
x = data.x.to(device)
edge_index = data.edge_index.to(device)
# Num of feat:1639
adj = to_scipy_sparse_matrix(edge_index = data.edge_index)
adj = adj+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#print(type(adj)) #<class 'scipy.sparse.csr.csr_matrix'>
#A_tilde = normalize_adjacency_matrix(adj,sp.eye(adj.shape[0]))
#adj_p = normalizemx(adj)
#
features = data.x
features = torch.FloatTensor(np.array(features))
features = features.to(device)
#model = GCN(nfeat=features.shape[1],nhid=64,nclass=10,nlayers=args.nlayers,dropout=args.dropout)

#labels = data.y
#labels = torch.LongTensor(np.array(labels))
#labels = labels.to(device)
#print(features.shape)
#print("========================")
#print('Loading')
#adj_sct1 = scattering1st(adj_p,1)
#adj_sct2 = scattering1st(adj_p,2)
#adj_sct4 = scattering1st(adj_p,3)
#adj_p = sparse_mx_to_torch_sparse_tensor(adj_p)
#A_tilde = sparse_mx_to_torch_sparse_tensor(A_tilde)
from diffusion import GCN_diffusion,scattering_diffusion
adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()
gcn_diffusion_list = GCN_diffusion(adj,3,features)
print(gcn_diffusion_list[1].size())
h_sct1,h_sct2,h_sct3 = scattering_diffusion(adj,features)
print(h_sct1.size())
model = SCT_GAT_ogbarxiv(features.shape[1],args.hid,10,dropout=args.dropout,nheads=args.nheads,smoo=args.smoo)
model = model.cuda()
out_feature = model(features,adj) 
print('=====')
print(out_feature.size())
