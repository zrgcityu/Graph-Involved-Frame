Implementation of GIF (Grpah-involved Frame). Please refer to [our paper](https://arxiv.org/abs/2309.03537) for details.

Run demo.py, denosing.py and nla.py for demos of thresholding-reconsturction, graph signal denosing and non-linear approximation.

The names of pickle files in folde 'train_filters' indicate the type of the base/frame, e.g basis_filter_1_20 = GIB(1,20), frame_1_20 = GIF(1,20).

Python packages required:

pymanopt, networkx, sortedcontainers, autograd, sknetwork, numpy, scipy.


