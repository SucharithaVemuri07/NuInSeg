# Parameter Efficient Fine-tuning Foundation Model for Nuclei Instance Segmentation

##Overview
This repository guides through the implementation of the finetuning Segment Anything Model for the downstream task - Nuclei Instance Segmentation. The NuInsSeg dataset @https://www.kaggle.com/datasets/ipateam/nuinsseg used to train and experiment with the model, the dataset consists of 31 human and mouse organs H&E Whole Slide Images. To work with this repository consider data folders from the main directory: Tissue Images and Label Masks. The block diagram provides a high-level interpretation of the work pipeline.
![Untitled Diagram drawio-19 drawio](https://github.com/user-attachments/assets/15548896-e905-4b06-bb88-d74547d511bb)




##Steps to Install Dependencies and Workflow
The model built and implemented on NVIDIA A100 GPU. The below steps are followed to work on GPU. If computing resource-limited, use CPU and reduce the batch_size. Create conda environment with required dependencies.
###Create Environment
