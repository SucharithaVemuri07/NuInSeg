# Parameter Efficient Fine-tuning Foundation Model for Nuclei Instance Segmentation

## Overview
This repository guides through the implementation of the finetuning Segment Anything Model for the downstream task - Nuclei Instance Segmentation. The NuInsSeg dataset @[Nuclei Instance Dataset](https://www.kaggle.com/datasets/ipateam/nuinsseg) used to train and experiment with the model, the dataset consists of 31 human and mouse organs H&E Whole Slide Images. To work with this repository consider data folders from the main directory: Tissue Images and Label Masks. The block diagram provides a high-level interpretation of the work pipeline.
![Block-Diagram](https://github.com/user-attachments/assets/15548896-e905-4b06-bb88-d74547d511bb)


## Requirements and Workflow
The model built and implemented on NVIDIA A100 GPU. The below steps are followed to work on GPU. If computing resources are limited (e.g., CPU-only), reduce the batch size in the config and work with the small variant. Create conda environment with required dependencies.

### Create Environment
```bash
conda create -n <env-name> python=3.11
conda activate <env-name>
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
## Model Setup
The pipeline is adapted and finetuned on the original Segment Anything Model and MobileSAM. Choose a model and a tiny variant to train. Download model checkpoints from the @[SAM](https://github.com/facebookresearch) for SAM checkpoints and @[MobileSAM](https://github.com/ChaoningZhang/MobileSAM) for Mobile SAM checkpoints.
Modify MODEL_CHECKPOINT_PATH in utils

### Install Segment Anything Model - SAM
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```
### Install Mobile Segment Anything Model - MobileSAM
```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```
Check the model website for more documentation on model installation and development.

### Training
Run 5-Fold Cross-Validation training:
```bash
python train.py
```

### Inference
Use the best-saved model to perform prediction. To use the pipeline only in on-inference mode, load the fold 4 best model from the saved_models directory:
```bash
python predict.py
```


## Acknowledgments
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation technique for efficient fine-tuning
- [FinetuneSAM](https://github.com/mazurowski-lab/finetune-SAM) - Different Finetuning techniques to finetune SAM on a custom medical dataset

Thanks to the open-source community!

If you have any questions or suggestions, feel free to reach out!

