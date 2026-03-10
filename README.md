# Paper ID: 10124
# GASegNet: Global Self-Attention Mechanism Meets Structural Feature Fusion for Point Cloud Semantic Segmentation

> Xu Lu, Haijun Liu, Guang'an Luo, Zhike Chen, Cheng Zhou, Xinyu Wu, Senior Member, IEEE, and Jun Liu

Xu Lu is with the School of Computer Science, Guangdong Polytechnic Normal University, Guangzhou 510665, China (e-mail: xulu@gpnu.edu.cn)
Jun Liu is with the School of Automation, Guangdong Polytechnic Normal University, Guangzhou 510665, China; (e-mail: liujun7700@163.com)
X.Wu is with the Shenzhen Institute of Advanced Technology, Shenzhen 518055, China.(e-mail: xy.wu@siat.ac.cn)
Haijun Liu, Guang'an Luo, Zhike Chen, and Cheng Zhou are with the Guangdong Polytechnic Normal University, Guangzhou 510665, China; (e-mail: haifeng@stu.gpnu.edu.cn; luoguangan@gpnu.edu.cn; chuck@gpnu.edu.cn; zhoucheng@gpnu.edu.cn).


## Abstract

With the rapid development of autonomous driving technology, semantic segmentation, as one of the key technologies contributing to the environment perception of autonomous driving systems, still suffers from a lack of connections between local features, as well as high computational consumption and an inability to meet real-time requirements. To address the above problems, this paper proposes a lightweight and efficient point cloud semantic segmentation network based on spherical projection with encoder-decoder structure. The encoder consists of a global self-attention mechanism that captures global information, as well as multi-scale convolution. This module achieves the unification of local feature extraction and global characteristic information for high-dimensional semantic information. In order to alleviate the high computational cost, a feature fusion module is introduced to enhance the compactness of the range image structure obtained from point cloud projection. The decoder utilizes bilinear interpolation to upsample multi-resolution feature maps, and introduces multiple auxiliary segmentation heads to further enhance the network's accuracy. Experiments conducted on the SemanticKITTI and SemanticPOSS datasets reveal that, in comparison to the CENet architecture, the proposed approach attains enhancements in mIoU of 4.3\% and 2.6\% on the respective datasets, thereby substantiating its efficacy. The code is available at GitHub: https://github.com/haifeng925/GASegNet.

## Key Features
- We propose GASegNet, a semantic segmentation network for LiDAR, which incorporates a Feature Fusion Module (FFM) and a Global Self-Attention Mechanism (GSAM) to enhance point cloud semantic segmentation.
- The Feature Fusion Module (FFM) is introduced to reconstruct geometric structural features of point cloud data and enhance the local feature aggregation capability of the network. This module effectively addresses the challenge of capturing fine-grained details in complex point cloud structures.
- The Global Self-Attention Mechanism (GSAM) is designed to focus on global information, enabling the network to capture higher-level semantic features. By capturing long-distance dependencies between any two locations, GSAM enhances the utilization of global contextual information and addresses the occlusion problem in point cloud data.


## Requirements

- Python 3.10 or later
- PyTorch (recommended >= 1.7, CUDA-enabled for GPU training)
- Common packages: numpy, tqdm, tensorboard



## Datasets

Supported datasets:
- SemanticKITTI — http://www.semantic-kitti.org/dataset.html
- SemanticPOSS — http://www.poss.pku.edu.cn./download.html

Organize datasets according to the original dataset structure so that the training and evaluation scripts can locate point clouds, labels, and sequence splits.

## Configuration

Model and training configurations are stored in `config/arch/`. Example config files:
- `config/arch/senet-512.yml` — example KITTI configuration
- `config/arch/poss.yml` — example SemanticPOSS configuration

Most scripts accept the `-ac` flag to load architecture/config YAML and `-n` to name the experiment.

## Description of Files:

The project is organized as follows:

```
GASegNet/
├── README.md                    # Project documentation (this file)
├── GASegNet.yml                 # Conda environment configuration file
│
├── GASegNet/                    # Main source code directory
│   ├── train.py                 # Main training script for SemanticKITTI
│   ├── train_poss.py            # Training script for SemanticPOSS dataset
│   ├── infer.py                 # Inference script for SemanticKITTI
│   ├── infer_poss.py            # Inference script for SemanticPOSS
│   ├── ts_infer.py              # TensorRT/ONNX inference script (optional)
│   ├── evaluate_iou.py          # Evaluation script to compute mIoU metrics
│   ├── visualize.py             # Visualization tool for point clouds and predictions
│   ├── kitti.sh                 # Shell script for SemanticKITTI training pipeline
│   ├── pos.sh                   # Shell script for SemanticPOSS training pipeline
│   ├── LICENSE                  # License file
│   │
│   ├── assert/                  # Assets and resources
│   │   ├── robustness.png       # Robustness evaluation results image
│   │   ├── rank.jpg             # Ranking visualization
│   │   ├── SENet.pdf            # Related paper/documentation
│   │   └── finalspeed.md        # Final speed benchmark documentation
│   │
│   ├── common/                  # Common utilities and data structures
│   │   ├── __init__.py
│   │   ├── laserscan.py         # LaserScan class for SemanticKITTI point cloud processing
│   │   ├── laserscanvis.py      # Visualization utilities for SemanticKITTI scans
│   │   ├── posslaserscan.py     # LaserScan class for SemanticPOSS point cloud processing
│   │   ├── posslaserscanvis.py  # Visualization utilities for SemanticPOSS scans
│   │   ├── avgmeter.py          # Average meter utility for tracking metrics
│   │   └── sync_batchnorm/      # Synchronized batch normalization implementation
│   │       ├── __init__.py
│   │       ├── batchnorm.py     # Synchronized BatchNorm2d implementation
│   │       ├── comm.py          # Communication utilities for multi-GPU sync
│   │       └── replicate.py    # Replication utilities
│   │
│   ├── config/                  # Configuration files
│   │   ├── arch/                # Architecture configuration files
│   │   │   ├── senet-512.yml    # GASegNet config for 64x512 resolution (KITTI)
│   │   │   ├── senet-1024p.yml # GASegNet config for 64x1024 resolution (KITTI)
│   │   │   ├── senet-2048p.yml  # GASegNet config for 64x2048 resolution (KITTI)
│   │   │   ├── poss.yml         # GASegNet config for SemanticPOSS dataset
│   │   │   └── poss-fid.yml     # FIDNet-based config for SemanticPOSS
│   │   └── labels/              # Label configuration files
│   │       ├── semantic-kitti.yaml        # SemanticKITTI label definitions
│   │       ├── semantic-kitti-all.yaml    # Extended SemanticKITTI labels
│   │       └── semantic-poss.yaml         # SemanticPOSS label definitions
│   │
│   ├── dataset/                 # Dataset loaders and parsers
│   │   ├── __init__.py
│   │   ├── kitti/              # SemanticKITTI dataset utilities
│   │   │   ├── __init__.py
│   │   │   └── parser.py       # SemanticKITTI data parser and loader
│   │   └── poss/               # SemanticPOSS dataset utilities
│   │       └── parser.py       # SemanticPOSS data parser and loader
│   │
│   ├── modules/                # Core modules and network components
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main training loop and logic for SemanticKITTI
│   │   ├── trainer_poss.py      # Training loop for SemanticPOSS
│   │   ├── user.py             # User interface class for SemanticKITTI (model loading, inference)
│   │   ├── user_poss.py        # User interface class for SemanticPOSS
│   │   ├── ioueval.py          # IoU evaluation utilities
│   │   │
│   │   ├── network/            # Network architecture implementations
│   │   │   ├── __init__.py
│   │   │   ├── ResNet.py       # ResNet-based backbone with GSAM and FFM modules
│   │   │   │                    # Contains: PAM_Module, CAM_Module, Fusion_Module,
│   │   │   │                    #          ScConv, SRU, CRU, ResNet_34
│   │   │   ├── HarDNet.py      # HarDNet backbone implementation
│   │   │   │                    # Contains: HarDBlock, HarDBlock_v2, HarDNet
│   │   │   └── Fid.py          # FIDNet-based architecture implementation
│   │   │                        # Contains: ResNet_34 (FIDNet variant)
│   │   │
│   │   ├── losses/             # Loss function implementations
│   │   │   ├── __init__.py
│   │   │   ├── Lovasz_Softmax.py    # Lovász-Softmax loss for semantic segmentation
│   │   │   └── boundary_loss.py     # Boundary-aware loss function
│   │   │
│   │   └── scheduler/          # Learning rate schedulers
│   │       ├── __init__.py
│   │       ├── adam_policy.py  # Adam optimizer learning rate policy
│   │       ├── cosine.py       # Cosine annealing scheduler
│   │       └── warmupLR.py     # Warmup learning rate scheduler
│   │
│   └── postproc/               # Post-processing utilities
│       ├── __init__.py
│       └── KNN.py              # K-Nearest Neighbors post-processing for refinement
```

### Key Components Description

#### Main Scripts
- **`train.py`**: Main training entry point for SemanticKITTI. Handles model initialization, data loading, training loop, checkpointing, and logging.
- **`train_poss.py`**: Similar to `train.py` but specifically designed for SemanticPOSS dataset with appropriate data loading and evaluation.
- **`infer.py`**: Performs inference on SemanticKITTI validation/test sets. Loads trained models and generates prediction files.
- **`infer_poss.py`**: Inference script for SemanticPOSS with built-in mIoU computation.
- **`evaluate_iou.py`**: Computes Intersection over Union (IoU) metrics for semantic segmentation predictions.
- **`visualize.py`**: Visualizes point cloud data, ground truth labels, and model predictions in 3D space.

#### Network Architectures (`modules/network/`)
- **`ResNet.py`**: Implements the main GASegNet architecture with:
  - **PAM_Module**: Position Attention Module for spatial attention
  - **CAM_Module**: Channel Attention Module for channel-wise attention
  - **Fusion_Module**: Feature fusion module combining attention outputs
  - **ScConv**: Spatial and Channel Convolution module
  - **SRU/CRU**: Spatial/Channel Recurrent Units
  - **ResNet_34**: Main ResNet-34 backbone with GSAM and FFM integration
- **`HarDNet.py`**: HarDNet backbone implementation, an alternative lightweight architecture.
- **`Fid.py`**: FIDNet-based architecture variant.

#### Data Processing (`common/` and `dataset/`)
- **`laserscan.py`**: Core class for processing SemanticKITTI LiDAR scans, including spherical projection, range image generation, and data augmentation.
- **`posslaserscan.py`**: Similar functionality for SemanticPOSS dataset.
- **`parser.py`**: Dataset-specific parsers that handle file I/O, label loading, and data organization.

#### Training Infrastructure (`modules/`)
- **`trainer.py`**: Comprehensive training manager handling:
  - Model initialization and optimization
  - Data loading and batching
  - Loss computation (cross-entropy, Lovász-Softmax, auxiliary losses)
  - Learning rate scheduling
  - Checkpoint saving/loading
  - TensorBoard logging
- **`user.py`**: Inference interface for loading models and running predictions.

#### Loss Functions (`modules/losses/`)
- **`Lovasz_Softmax.py`**: Lovász-Softmax loss, effective for imbalanced class distributions in semantic segmentation.
- **`boundary_loss.py`**: Boundary-aware loss to improve segmentation quality at object boundaries.

#### Post-processing (`postproc/`)
- **`KNN.py`**: K-Nearest Neighbors post-processing to refine predictions using spatial consistency.

## Training

### SemanticKITTI

Recommended staged training strategy (used in our experiments to balance GPU/time constraints):
1. Train on low resolution (e.g., 64x512).
2. Load the pre-trained weights and train on medium resolution (e.g., 64x1024).
3. Optionally fine-tune on higher resolution (e.g., 64x2048).

Example command:

```bash
python train.py -d /path/to/semantic_kitti -ac config/arch/senet-512.yml -n senet-512
```

If you want to resume training from a checkpoint, see the commented guidance in `modules/trainer.py` for how to restore from saved states.

### SemanticPOSS

Example command:

```bash
python train_poss.py -d /path/to/semanticposs -ac config/arch/poss.yml -n poss-exp
```

## Inference and Evaluation

### SemanticKITTI

Run inference (valid or test):

```bash
python infer.py -d /path/to/semantic_kitti -l /path/to/output_predictions -m /path/to/trained_model -s valid
```

Evaluate predictions (validation):

```bash
python evaluate_iou.py -d /path/to/semantic_kitti -p /path/to/output_predictions
```

For test sequences, prepare predictions according to Benchmark instructions and submit via CodaLab.

### SemanticPOSS

Run inference and compute mIoU:

```bash
python infer_poss.py -d /path/to/semanticposs -l /path/to/output_predictions -m /path/to/trained_model
```

This script will produce prediction files and report mIoU on the provided split.

## Visualization

Visualize ground truth:

```bash
python visualize.py -w kitti/poss -d /path/to/dataset -s sequence_ids
```

Visualize predictions:

```bash
python visualize.py -w kitti/poss -d /path/to/dataset -p /path/to/predictions -s sequence_ids
```

Adjust `-w` and other flags according to available visualization backends in the repo.


## Troubleshooting & Tips
- If your segmentation performance is lower than expected, check training hyperparameters, input resolution, data augmentation, and whether you used the same pre-processing as in the original experiments.
- For limited GPU memory, follow the staged training strategy (low → high resolution).
- If data paths are mismatched, examine dataset loader code and adjust path parsing to your local layout.



## Acknowledgements

This repository builds upon implementations and ideas from:
- SalsaNext
- FIDNet
- SqueezeSegV3

Thanks to the authors of these projects for sharing their code and to colleagues who provided helpful discussions.

## Contact

For questions or replication of results: haifeng@stu.gpnu.edu.cn, luoguangan@gpnu.edu.cn.

