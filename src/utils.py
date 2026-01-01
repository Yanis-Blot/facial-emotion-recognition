import torch
from config import device
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from config import NUM_CLASSES, device


# Train step function
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):

    train_loss, train_acc = 0, 0

    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device) # On se met sur le device actuel (si CUDA, X sera envoyé sur le GPU)

        # Etape 1 : Forward Pass
        y_pred = model(X)

        # Etape 2 : on calcule la perte et l'accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss

        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred)

        # Etape 3 : on remet a zéro l'optimizer
        optimizer.zero_grad()

        # Etape 4 : backpropagation sur la graphe de calcul
        loss.backward()

        # Etape 5 : Application de l'optimizer (SGD)
        optimizer.step()

    # Perte et Accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    return train_loss, train_acc

# Test step function :
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)

    #On passe en mode d'evaluation pour accélérer le temps de calcul au test
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred = model(X)

            # Calcul de la perte de accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred
            )

        #Normalisation des valeurs (-> on fait une moyenne d'accuracy et de perte sur le batch)
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        return test_loss, test_acc

# Accuracy function
def accuracy_fn(y_pred, y_true):
    """Calcule l'accuracy sur un batch"""
    y_pred_classes = torch.argmax(y_pred, dim=1)
    correct = (y_pred_classes == y_true).sum().item()
    acc = correct / len(y_true)
    # Return accuracy as a percentage
    return 100 * acc

# Display accuracy and loss
def display_metrics(losses, accuracies):
    # Préparation de la figure
    epochs = range(1, len(accuracies) + 1)

    plt.figure(figsize=(12, 6))

    # Plot de la loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Losses" )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)

    # Plot de l'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Accuracies" )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Confusion Matrix function
def compute_confmat(model, dataloader, device = device):
    confmat = ConfusionMatrix(
        task="multiclass",
        num_classes=NUM_CLASSES
    ).to(device)

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)
            y_preds = y_logits.argmax(dim=1)

            confmat.update(y_preds, y)

    return confmat.compute().cpu()

