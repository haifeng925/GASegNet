# GASegNet

Global Self-Attention Mechanism Meets Structural Feature Fusion for Point Cloud Semantic Segmentation

> Code accompanying the paper "GASegNet: Global Self-Attention Mechanism Meets Structural Feature Fusion for Point Cloud Semantic Segmentation"

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

For issues, feature requests or questions, please open an issue on the repository or contact the maintainers.

