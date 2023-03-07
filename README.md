# A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?

This is the official repository for the paper "[A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?](https://arxiv.org/abs/2302.11640)" (ICLR 2023).

### Datasets

In the paper we introduce 5 new heterophilous graph datasets: roman-empire, amazon-ratings, minesweeper, tolokers, and questions. You can find these datasets in the `data` folder. Note: the datasets are stored as undirected, that is, each edge is stored only once. If you load the edges into DGL or PyG, which treat all graphs as directed, do not forget to call `dgl.to_bidirected` or `pyg.transforms.ToUndirected` to double the edges.

Roman-empire is a word dependency graph based on the Roman Empire article from the [English Wikipedia](https://huggingface.co/datasets/wikipedia).

Amazon-ratings is a product co-purchasing network based on data from [SNAP Datasets](https://snap.stanford.edu/data/amazon-meta.html).

Minesweeper is a synthetic graph emulating the eponymous game.

Tolokers is a crowdsourcing platform workers network based on [data](https://github.com/Toloka/TolokerGraph) provided by [Toloka](https://toloka.ai).

Questions is an interaction graph of users of a question-answering website based on data provided by [Yandex Q](https://yandex.ru/q).

Our datasets come from different domains and exhibit a wide range of structual properties. We provide some statistics of our datasets in the table below:

|                       | roman-empire | amazon-ratings | minesweeper | tolokers | questions |
|-----------------------|:------------:|:--------------:|:-----------:|:-------:|:---------:|
| nodes                 |     22662    |      24492     |    10000    |  11758  |   48921   |
| edges                 |     32927    |      93050     |    39402    |  519000 |   153540  |
| avg degree            |     2.91     |      7.60      |     7.88    |  88.28  |    6.28   |
| global clustering     |     0.29     |      0.32      |     0.43    |   0.23  |    0.02   |
| avg local clustering  |     0.39     |      0.58      |     0.44    |   0.53  |    0.03   |
| diameter              |     6824     |       46       |      99     |    11   |     16    |
| node features         |      300     |       300      |      7      |    10   |    301    |
| classes               |      18      |        5       |      2      |    2    |     2     |
| edge homophily        |     0.05     |      0.38      |     0.68    |   0.59  |    0.84   |
| adjusted homophily    |     -0.05    |      0.14      |     0.01    |   0.09  |    0.02   |
| label informativeness |     0.11     |      0.04      |     0.00    |   0.01  |    0.00   |

More details on our datasets can be found in the paper.

### Reproducing experiments

In our paper we show that standard GNNs augmented with skip connections and layer normalization achieve strong results on heterophilous graphs and almost always outperform specialized models.

To reproduce results of our baseline models (ResNet and standard GNNs), you need to install [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/pages/start.html) (see the full list of requirements in `enivronment.yml`). Then you can run `scripts/run_all_experiments.sh` (there are a lot of experiments, so you might want to comment out some of the lines). After that you can view a table with results in `notebooks/results.ipynb`.

To reproduce results of specialized models, see [this repository](https://github.com/Godofnothing/HeterophilySpecificModels).

### Citation

If you found our datasets or our code helpful, please cite our paper:

```buildoutcfg
@inproceedings{platonov2023critical,
  title={A critical look at evaluation of GNNs under heterophily: Are we really making progress?},
  author={Platonov, Oleg and Kuznedelev, Denis and Diskin, Michael and Babenko, Artem and Prokhorenkova, Liudmila},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```
