# Semantic Graph Counterfactuals

Official Repository for "Structure Your Data: Towards Semantic Graph Counterfactuals" accepted at ICML 2024 ([paper link](https://arxiv.org/abs/2403.06514)).

Our paper addresses conceptual counterfactual explanations by structuring data as graphs and calculating counterfactual instances via graph neural network representations.

<p align = "center"><img src = "https://github.com/aggeliki-dimitriou/SGCE/blob/main/outline.png"></p><p align = "center">
  <b>Fig.1</b>: Overview of our system
</p>

## Cite

```
@article{dimitriou2024structure,
  title={Structure Your Data: Towards Semantic Graph Counterfactuals},
  author={Dimitriou, Angeliki and Lymperaiou, Maria and Filandrianos, Giorgos and Thomas, Konstantinos and Stamou, Giorgos},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```

## Installing requirements
- Create a new conda environment by running `conda create --name myenv --file requirements.txt`

## Ground truth construction
Code is available in 'ground_truth' directory
- Run <python main.py> as it is to produce GEDS for VG-DENSE. 
To run for other datasets: change paths of variables 'large_graphs', 'large_graphs_idx' 
so that they point to appropriate pickle files.

## Training GNN
Code is available in 'gnns/approach_1' directory.
- Make output directory if non-existent (i.e. in code/ directory, run <mkdir outs>)
- Make sure you have generated GED pairs to use as labels during similarity-based training
- Properly modify 'config.py', comments are available next to parameters
- Run `python main.py`. After training embeddings and the trained model are dumped in output dir

## Graph kernels
Code is available in 'graph_kernels' directory.

- Change paths in lines marked with '###' if running on different data subset
- Run `python kernels.py`

## Evaluation
Code is available in 'evaluation' directory
- Change variables 'new_idxs', 'vg_dense_classification', 'geds' in 'eval.py' according to data subset in use
- Provide paths to embeddings or graph kernel matrices (i.e. populate lists 'sims_rank', 'embedding_res')
  Comment out lines if you are not running full evaluation.
  The currect script evaluates best GNN embedding models for the 3 variants and 5 graph kernels.
- Run `python eval.py`

## Graph construction
Code for the construction of graphs for experiments (D/P-SGG, D/P-CAPTION, SMARTY, AG) is available in 'graph-extraction' directory.

Data for VG-DENSE use case can be found in 'data' directory. 'glove_emb_300.pkl' can be found [here](https://drive.google.com/file/d/1km1BFN3R0cQS2hWcOgRxz-J5GE3bK7DX/view?usp=drive_link) .



