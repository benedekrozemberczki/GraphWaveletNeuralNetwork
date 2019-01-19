Graph Wavelet Neural Network
============================================
A PyTorch implementation of "Graph Wavelet Neural Network" (ICLR 2019)
<div style="text-align:center"><img src ="attentionwalk.jpg" ,width=720/></div>
<p align="justify">
We present graph wavelet neural network (GWNN), a novel graph convolutional neural network (CNN), leveraging graph wavelet transform to address the shortcomings of previous spectral graph CNN methods that depend on graph Fourier transform. Different from graph Fourier transform, graph wavelet transform can be obtained via a fast algorithm without requiring matrix eigendecomposition with high computational cost. Moreover, graph wavelets are sparse and localized in vertex domain, offering high efficiency and good interpretability for graph convolution. The proposed GWNN significantly outperforms previous spectral graph CNNs in the task of graph-based semi-supervised classification on three benchmark datasets: Cora, Citeseer and Pubmed.</p>

This repository provides an implementation of Attention Walk as described in the paper:

> Graph Wavelet Neural Network.
> Bingbing Xu, Huawei Shen, Qi Cao, Yunqi Qiu, Xueqi Cheng.
> ICLR, 2019.
> [[Paper]](https://openreview.net/forum?id=H1ewdiR5tQ)

### Requirements

The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             0.4.1
torch-scatter     1.0.4
torch-sparse      0.2.2
torchvision       0.2.1
```
### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. Sample graphs for the `Twitch Brasilians` and `Wikipedia Chameleons` are included in the  `input/` directory. 

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --edge-path         STR   Input graph path.     Default is `input/chameleon_edges.csv`.
  --embedding-path    STR   Embedding path.       Default is `output/chameleon_AW_embedding.csv`.
  --attention-path    STR   Embedding path.       Default is `output/chameleon_AW_attention.csv`.
```

#### Model options

```
  --dimensions              INT       Number of embeding dimensions.        Default is 128.
  --epochs                  INT       Number of training epochs.            Default is 200.
  --window-size             INT       Skip-gram window size.                Default is 5.
  --learning-rate           FLOAT     Learning rate value.                  Default is 0.01.
  --beta                    FLOAT     Regularization parameter.             Default is 0.1.
```

### Examples

The following commands learn a graph embedding and write the embedding to disk. The node representations are ordered by the ID.

Creating a Attention Walk embedding of the default dataset with the default hyperparameter settings. Saving the embedding at the default path.

```
python src/main.py
```
<p align="center">
<img style="float: center;" src="attention_walk_run_example.jpg">
</p>

Creating an Attention Walk embedding of the default dataset with 256 dimensions.

```
python src/main.py --dimensions 256
```

Creating an Attention Walk embedding of the default dataset with a higher window size.

```
python src/main.py --window-size 20
```

Creating an embedding of an other dataset the `Twitch Brasilians`. Saving the outputs under custom file names.

```
python src/main.py --edge-path input/ptbr_edges.csv --embedding-path output/ptbr_AW_embedding.csv --attention-path output/ptbr_AW_attention.csv
```
