import torch
from config import WEIGHTS_DIR, METRICS_DIR, device
from utils import compute_confmat, display_metrics
from models import EmotionCNN
from dataset import load_datasets, get_dataloader
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os

# -- Parameters -- init the .csv file corresponding to the model you want to evaluate
METRICS_FILE = "2026-01-01_14-32-45_EmotionCNN_metrics_lr0.1_bs32" # Paste file name here
WEIGHTS_FILE = "model_final_lr0.1_bs32"



model = EmotionCNN(input_shape=1, hidden_units=128, output_shape=6).to(device)
model_name = model.__class__.__name__
model.load_state_dict(torch.load(os.path.join(f"{WEIGHTS_DIR}/{WEIGHTS_FILE}.pth"), map_location=device))

train_ds, test_ds, class_names = load_datasets()
train_loader = get_dataloader(train_ds)

# -- Init csv --
METRICS_PATH = f"{METRICS_DIR}/{METRICS_FILE}.csv"

# -- Read csv --
df = pd.read_csv(METRICS_PATH)

# -- Train --
# -- Extract loss and accuracy --
losses = df["train_loss"].tolist()
accuracies = df["train_acc"].tolist()

display_metrics(losses, accuracies)


cm = compute_confmat(model, dataloader=train_loader, device=device)
fig, ax = plot_confusion_matrix(
    conf_mat=cm.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)
plt.show()