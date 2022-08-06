# Baseline code for ScatteringAttention
Our code is based on https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/gnn.py
## Installation requirements
```
torch-geometric>=1.6.0
ogb version == 1.3.3
torch version=1.11.0
```


### ScatteringAttention (hid=30)
```
python RunScattering.py --dropout 0. --num_layers 2  --smoo 1.0 --hidden_channels 30 --runs 3 --lr 1e-2 --epochs 1500
```

## Measuring the Training Time
| Model              |Highest Train  | Highest Valid  | \#Parameters | Final Train | Final Test | Hardware |
|:------------------ |:--------------|:-----------------|:--------------|:----------|:--------------|-------------------|
| ScatteringAttention (hid=30) | 84.16 ± 0.25 | 80.54 ± 0.30| 3,832 | 84.14 ± 0.30  |73.93 ±  0.19|NVIDIA Tesla V SXM2 (32G) |


## References
1. Min, Y., Wenkel, F. and Wolf, G., 2020. Scattering gcn: Overcoming oversmoothness in graph convolutional networks. Advances in Neural Information Processing Systems, 33, pp.14498-14508.

2. Min, Y., Wenkel, F. and Wolf, G., 2021, June. Geometric scattering attention networks. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 8518-8522). IEEE.

3. Wenkel, F., Min, Y., Hirn, M., Perlmutter, M. and Wolf, G., 2022. Overcoming Oversmoothness in Graph Convolutional Networks via Hybrid Scattering Networks. arXiv preprint arXiv:2201.08932.

