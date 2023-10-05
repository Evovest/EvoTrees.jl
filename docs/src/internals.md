
# Internal API

## General

```@docs
EvoTrees.TrainNode
EvoTrees.EvoTree
EvoTrees.check_parameter
EvoTrees.check_args
```

## Training utils

```@docs
EvoTrees.init
EvoTrees.grow_evotree!
EvoTrees.update_gains!
EvoTrees.predict!
EvoTrees.subsample
EvoTrees.split_set_chunk!
EvoTrees.split_chunk_kernel!
```

## Histogram

```@docs
EvoTrees.get_edges
EvoTrees.binarize
EvoTrees.update_hist!
EvoTrees.hist_kernel!
EvoTrees.hist_kernel_vec!
EvoTrees.predict_kernel!
```
