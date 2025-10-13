
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
EvoTrees.get_best_split
EvoTrees.update_gains!
EvoTrees.predict!
EvoTrees.subsample
EvoTrees.split_set_chunk!
```

## Histogram

```@docs
EvoTrees.get_edges
EvoTrees.binarize
EvoTrees.update_hist!
```

## GPU Extension (CUDA)

### Main Functions

```@docs
EvoTreesCUDAExt.grow_tree!
EvoTreesCUDAExt.grow_otree!
EvoTreesCUDAExt.update_hist_gpu!
```

### GPU Kernels

```@docs
EvoTreesCUDAExt.update_nodes_idx_kernel!
EvoTreesCUDAExt.hist_kernel!
EvoTreesCUDAExt.reduce_root_sums_kernel!
EvoTreesCUDAExt.find_best_split_from_hist_kernel!
EvoTreesCUDAExt.apply_splits_kernel!
EvoTreesCUDAExt.clear_hist_kernel!
EvoTreesCUDAExt.clear_mask_kernel!
EvoTreesCUDAExt.mark_active_nodes_kernel!
```

