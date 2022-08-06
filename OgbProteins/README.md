#Baseline code for ScatteringAttention
Our code is based on https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/gnn.py
## Installation requirements
```
torch-geometric>=1.6.0
ogb version == 1.3.3
torch version=1.11.0
```


### ScatteringAttention (hid=30)
```
python -u RunScattering.py --dropout 0. --num_layers 2  --smoo 1.0 --hidden_channels 32 --runs 3 --lr 1e-2 --epochs 2000
```

## Measuring the Training Time


| Model              |Highest Train  | Highest Valid  | \#Parameters    | Final Train:| Final Test:| Hardware |
|:------------------ |:--------------   |:---------------| --------------:|----------|--------------:|---------------:|
| ScatteringAttention (hid=30)     | 84.16 ± 0.25 | 80.98 ± 0.15 3,83 84.14 ± 0.3|73.93 ± 0.19|NVIDIA Tesla VSXM2(32 GB)|


