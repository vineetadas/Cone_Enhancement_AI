# Description

This is a Pytorch implementation of residual in residual transformer generative adversarial network (RRTGAN), an artificial intelligence (AI) method  for enhancing cone visualization from sparsely sampled adaptive optics optical coherence tomography (AOOCT) images. 

If any portion of this code is used, please cite the following paper in your publication:

V. Das et al. "Artificial Intelligence Assisted Adaptive Optics Imaging Enables Dense Pixel Sampling from Sparse Measurements". 

# System Requirements

### Prerequisites

- The method was trained and tested on a A100 GPU server. 4 NVIDIA A100 GPUs with CUDA 11.8 was used.

### Installation

- Install [Anaconda](https://www.anaconda.com/products/distribution)
- In the anaconda prompt:
```
conda create -n <newenv> --file <path to the requirements.txt file provided in this repository>
```

# File Description

- A demo test dataset has been deposited in `./data/test_images`. The folder contains one example sparsely sampled image which are input to the model (RRTGAN) `(./data/test_data/input)`. 

- The ground truth image and the RRTGAN recovered image are provided in `(./data/test_images/groundtruth)` and `(./data/test_images/result)`, respectively.

- The trained model weights are deposited in `./data/trained_model`.

- The python files `train_model.py` and `python test_model.py` contain the PyTorch implementation of the training and testing pipelines for RRTGAN.

- `models.py`  defines the RRTGAN architecture.

# Demo

### Test

- Run `python test_model.py` to test the trained model whose weights are
  deposited in `./data/trained_model`.
- The results are stored in `./data/test_images/result`.

### Train

- To train the model on custom data, run 
```
python train_model.py --train_dir <path to training data> --path_train_data_fname <path to a .txt file containing the file names of the training images> --model_dir <path to folder for saving the trained model>
```

