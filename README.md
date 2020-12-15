<img src="./tab.png" width="250px"></img>

## Tab Transformer (wip)

Implementation of <a href="https://arxiv.org/abs/2012.06678">Tab Transformer</a>, attention network for tabular data, in Pytorch

## Install

```bash
$ pip install tab-transformer-pytorch
```

## Usage

```python
import torch
from tab_transformer_pytorch import TabTransformer

model = TabTransformer(
    num_unique_categories = 100,    # sum of unique categories across all categories
    num_categories = 10,            # number of categorical values
    num_continuous = 10,            # number of continuous values
    dim = 32,                       # dimension, paper set at 32
    dim_out = 1,                    # binary prediction, but could be anything
    depth = 6,                      # depth, paper recommended 6
    heads = 8                       # heads, paper recommends 8
)

x_categ = torch.randint(0, 100, (1, 10))  # categorical values given a unique id across all categories
x_cont = torch.randn(1, 10)               # assume continuous values are already normalized individually

pred = model(x_categ, x_cont)
```
## Citations

```bibtex
@misc{huang2020tabtransformer,
    title={TabTransformer: Tabular Data Modeling Using Contextual Embeddings}, 
    author={Xin Huang and Ashish Khetan and Milan Cvitkovic and Zohar Karnin},
    year={2020},
    eprint={2012.06678},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
