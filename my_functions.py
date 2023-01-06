import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from tqdm import tqdm
from timeit import default_timer as timer
import random
from torchvision.transforms.functional import pil_to_tensor
import mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
#def import_matrix():
#  #try:
 #   import torchmetrics, mlxtend
  #  print(f"mlxend version: {mlxtend.__version__}")
   # assert int(mlxtend.__version__.split(".")[1] >= 19, "mlxtend version should be 0.19.0 or higher.")
  #except:
   # !pip install -q torchmetrics -U mlxtend
   # import torchmetrics, mlxtend
   # print(f"mlxtend version: {mlxtend.__version__}")


def show_data(data):
    labels_map = data.classes

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label], c="g")
        plt.imshow(img.squeeze(), cmap="gray")
        plt.axis(False)
    plt.show()


def iter_dataloader(data_loader: torch.utils.data.DataLoader):
    train_features, train_labels = next(iter(data_loader))
    print(f"Feature batch Shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze().permute(1, 2, 0)
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.axis(False)
    plt.show()
    print(f"label: {label}")
    
    
def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred)) * 100
  return acc


def train_loop(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """Performs a training step with model trying to learn on data_loader """
    train_loss, train_acc = 0, 0

    # Put model into training mode
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)
        # Forward pass -> outputs raw logits
        y_pred = model(X)
        # Calculate the loss and accuracy per batch
        loss = loss_fn(y_pred,
                       y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))  # go from logits -> prediction labels using argmax()
        # Zero gradients
        optimizer.zero_grad()
        # Back propagation
        loss.backward()
        # step the optimizer (once per batch)
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


# Testing loop

def test_loop(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """Performs a testing loop step on model going over data_loader"""

    test_loss, test_acc = 0, 0
    # Put model into evaluation mode
    model.eval()
    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to target device
            X, y = X.to(device), y.to(device)
            # Forward pass -> outputs raw logits
            test_pred = model(X)
            # Calculate the test loss and accuracy per batch
            loss = loss_fn(test_pred,
                           y)
            test_loss += loss
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))  # go from logits -> prediction labels using armax()

        # Divide total test loss and acc by length of train dataloader
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%")


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.nn.Module,
               device: torch.device):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    optimizer = optimizer
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Make our data device agnostic
            X, y = X.to(device), y.to(device)
            # Make predictions
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return {"model_name": model.__class__.__name__,
            "loss_fn": loss_fn.__class__.__name__,
            "optimzer": optimizer.__class__.__name__,
            "model_loss": test_loss,
            "model_acc": f"{(100 * correct):.1f}%"
            }


def plot_matrix(model: nn.Module,
                data: DataLoader,
                device: torch.device = device):
    targets = test_data.targets
    class_names = test_data.classes
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data, desc="Making predictions.."):
            X, y, = X.to(device), y.to(device)
            y_logit = model(X)
            pred_prob = nn.Softmax(dim=0)(y_logit)
            y_pred = pred_prob.argmax(1)
            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)

    confmat = ConfusionMatrix(task="multiclass",
                              num_classes=len(class_names))
    confmat_tensor = confmat(preds=y_pred_tensor,
                             target=targets)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7)
    )


def train_time(start: int,
               end: int,
               device: torch.device = None):
  """Prints difference between start and end time."""
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time


def make_predictions(model: torch.nn,
                       data,
                       device: torch.device = device):

  pred_probs = []
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample, dim=0).to(device)
      pred_logit = model(sample)
      pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
      pred_probs.append(pred_prob.cpu())
  return torch.stack(pred_probs)


def plot_preds(model: nn.Module,
               data,
               device: torch.device = device):
    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = make_predictions(model=model,
                                  data=test_samples,
                                  device=device)

    pred_classes = pred_probs.argmax(dim=1)
    class_names = train_data.classes

    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in tqdm(enumerate(test_samples)):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(sample.squeeze(), cmap="gray")
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        title_text = f"Pred:{pred_label} | Truth: {truth_label}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")
        plt.axis(False);



