import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 1024
BATCH_SIZE_TRAIN = 3
BATCH_SIZE_VAL = 2
NUM_FOLDS = 5
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

save_dir = "saved_models/"
MODEL_CHECKPOINT_PATH = "path/to/model_checkpoint"
