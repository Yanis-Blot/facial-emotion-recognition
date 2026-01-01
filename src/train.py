import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
from config import INPUT_SHAPE, HIDDEN_UNITS, NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, MANUAL_SEED, METRICS_DIR, WEIGHTS_DIR, device
from models import EmotionCNN
from dataset import load_datasets, get_dataloader
from utils import accuracy_fn, train_step, test_step
from datetime import datetime
import csv

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

torch.manual_seed(MANUAL_SEED)

# -- Dataloader --
train_ds, test_ds, class_names = load_datasets(True)
train_loader = get_dataloader(train_ds)
test_loader = get_dataloader(test_ds)


# -- Init losses and accuracies --
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


# -- Model --
model = EmotionCNN(input_shape=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, output_shape=NUM_CLASSES).to(device)
model_name = model.__class__.__name__
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=25, gamma=0.75)


# -- Init csv --
METRICS_PATH = os.path.join(
    METRICS_DIR,
    f"{RUN_ID}_{model_name}_lr{LEARNING_RATE}_bs{BATCH_SIZE}.csv")

with open(METRICS_PATH, "w", newline="") as f: # Metrics saved in models/metrics
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc"
    ])


# -- Train and test--
epochs = EPOCHS
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch+1}\n---------")

    # -- Training --
    train_loss, train_acc = train_step(data_loader=train_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device)
    
    scheduler.step()

    print(f"Train loss: {train_loss} | Train accuracy: {train_acc}%")
    train_losses.append(train_loss.cpu().item())
    train_accuracies.append(train_acc)



    # -- Test --
    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = test_step(model, test_loader, loss_fn, accuracy_fn, device)

        print(f"Test loss: {test_loss} | Test accuracy: {test_acc}%\n")
        test_losses.append(test_loss.cpu().item())
        test_accuracies.append(test_acc)


    # Append data in the csv
    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss.cpu().item(),
            train_acc,
            test_loss.cpu().item(),
            test_acc
        ])

print("Training Successful")

# Model saved in models/
torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"{RUN_ID}_{model_name}_lr{LEARNING_RATE}_bs{BATCH_SIZE}.pth"))