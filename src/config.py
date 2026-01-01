# Centralize Hyperparameters, path and device informations

import torch

# -- device agnostic code --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Hyperparameters --
# Reproducibility
MANUAL_SEED= 42

# Model
INPUT_SHAPE = 1
HIDDEN_UNITS = 128
NUM_CLASSES = 6

# Training
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.1


# -- Pathes --
DATA_DIR = "data/"
TRAIN_DIR = "data/train/"
TEST_DIR = "data/test/"
MODEL_DIR = "src/models"
WEIGHTS_DIR = "outputs/weights"
METRICS_DIR = "outputs/metrics"
