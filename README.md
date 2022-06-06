# Towards Better Evaluation for Dynamic Link Prediction



## Introduction

Recently, increased attention has been given to the development of novel dynamic graph representation learning methods. 
In this work,
we revisit current evaluation settings for link prediction on dynamic graphs. 
Using two novel visualization techniques for edge statistics in dynamic graphs, we observe that a large portion of edges in dynamic graphs naturally reoccur over time, and recurrence patterns vary significantly across datasets.

Based on these observations and motivated by real-world applications, we first propose two novel negative sampling strategies for evaluation of link prediction in dynamic graphs. 
Performance of existing methods degrades significantly when the set of negative edges used during evaluation is chosen more selectively. 
This shows that it is necessary to conduct different negative sampling strategies beyond simple random sampling to fully understand the performance of a given method. 
Second, we propose a simple baseline, EdgeBank, solely based on memorizing past edges. 
We gain surprisingly strong performance with EdgeBank across multiple settings and we suggest that future methods should consider comparing their performance against this simple memorization approach. 
Lastly, we introduce five new dynamic networks from novel graph domains not present in the current benchmarks. 
These datasets offer new challenges for link prediction methods and enable a more robust evaluation setup.



## Running the experiments

### Requirements
* `python >= 3.7`, `PyTorch >= 1.4`
* Other requirements:

```{bash}
pandas==1.1.0
scikit_learn==0.23.1
tqdm==4.41.1
numpy==1.16.4
matploblib==3.3.1
```

#### Datasets and Processing
All dynamic graph datasets can be downloaded from [here](https://drive.google.com/file/d/1EGmV_js2DzocxwArhO12rizqPY2nZjSf/view?usp=sharing).
Then, they can be located in *"DG_data"* folder.
For conducting any experiments, the required data should be in the **data** folder under each model of interest.
* For example, to train a *TGN* model on *Wikipedia* dataset, we can use the following command to move the edgelist to the right folder:
```{bash}
cp DG_data/wikipedia.csv tgn/data/wikipedia.csv
```

* Then, the edgelist should be pre-processed to have the right format.
Considering the example of *Wikipedia* edgelist, we can use the following command for pre-processing the dataset:
```{bash}
# JODIE, DyRep, or TGN
python tgn/utils/preprocess_data.py --data wikipedia

# TGAT
python tgat/preprocess_data.py --data wikipedia

# CAWN
python CAW/preprocess_data.py --data wikipedia
```


### Model Training
* Example of training different graph representation learning models on *Wikipedia* dataset:
```{bash}
data=wikipedia
n_runs=5

# JODIE
method=jodie
prefix="${method}_rnn"
python tgn/train_self_supervised.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0

# DyRep
method=dyrep
prefix="${method}_rnn"
python tgn/train_self_supervised.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0

# TGN
method=tgn
prefix="${method}_attn"
python tgn/train_self_supervised.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0

# TGAT
prefix="TGAT"
python -u tgat/learn_edge.py -d $data --bs 200 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix $prefix --n_runs $n_runs --gpu 0

# CAWN
python CAW/main.py -d $data --pos_dim 108 --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --n_runs "$n_runs" --gpu 0

```

* Example of using EdgeBank for dynamic link prediction with standard *random* negative sampler:
```{bash}
data=Wikipedia
mem_mode=unlim_mem
n_runs=5
neg_sample=rnd  # can be one of these options: "rnd": standard randome, "hist_nre": Historical NS, or "induc_nre": Inductive NS
python EdgeBank/link_pred/edge_bank_baseline.py --data "$data" --mem_mode "$mem_mode" --n_runs "$n_runs" --neg_sample "$neg_sample"
```

### Testing Trained Models
* Testing trained models with different negative edge sampling strategies:
```{bash}
n_runs=5
data=wikipedia
neg_sample=hist_nre  # can be either "hist_nre" for historical NS, or "induc_nre" for inductive NS

# JODIE
method=jodie
python tgn/tgn_test_trained_model_self_sup.py -d "$data" --use_memory --memory_updater rnn --embedding_module time --model $method --neg_sample $neg_sample --n_runs $n_runs --gpu 0

# DyRep
method=dyrep
python tgn/tgn_test_trained_model_self_sup.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --model $method --neg_sample $neg_sample --n_runs $n_runs --gpu 0

# TGN
method=tgn
python tgn/tgn_test_trained_model_self_sup.py -d $data --use_memory --model $method --neg_sample $neg_sample --n_runs $n_runs --gpu 0

# TGAT
prefix="TGAT"
python tgat/tgat_test_trained_model_learn_edge.py -d $data --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix $prefix --n_runs $n_runs --neg_sample $neg_sample

# CAWN
python CAW/caw_test_trained_model_main.py -d $data --pos_dim 108 --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --n_runs "$n_runs" --gpu 0 --neg_sample $neg_sample

```

### Visualizing Dynamic Graphs
For visualizing the dynamic networks, their edgelists should be located in the *"visualization/data/"* folder.

* For generating **_TEA_** plots:
```{bash}
python visualization/TEA_plots.py
```
(Different networks can be selected directly in the *"visualization/TEA_plots.py"* file.)

* For generating **_TET_** plots:
```{bash}
python visualization/TET_plots.py
```
(Different networks can be selected directly in the *"visualization/TET_plots.py"* file.)

The outputs are saved in *"visualization/figs/TEA"* or *"visualization/figs/TET"* folder for the *TEA* or *TET* plots, respectively.

### Acknowledgment
We would like to thanks the authors of [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs), [TGN](https://github.com/twitter-research/tgn), and [CAWN](https://github.com/snap-stanford/CAW) projects for providing access to their projects' code.

