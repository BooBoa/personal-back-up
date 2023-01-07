import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
import zipfile
from pathlib import Path
from typing import *
import requests
from PIL import Image
import pathlib
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

def plot_transformed_images(image_paths, transform, n=3, seed=42):
     """Plots a series of random images from image_paths.

     Will open n image paths from image_paths, transform them
     with transform and plot them side by side.

     Args:
         image_paths (list): List of target image paths. 
         transform (PyTorch Transforms): Transforms to apply to images.
         n (int, optional): Number of images to plot. Defaults to 3.
         seed (int, optional): Random seed for the random generator. Defaults to 42.
     """
     random.seed(seed)
     random_image_paths = random.sample(image_paths, k=n)
     for image_path in random_image_paths:
         with Image.open(image_path) as f:
             fig, ax = plt.subplots(1, 2)
             ax[0].imshow(f) 
             ax[0].set_title(f"Original \nSize: {f.size}")
             ax[0].axis("off")

             # Transform and plot image
             # Note: permute() will change shape of image to suit matplotlib 
             # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
             transformed_image = transform(f).permute(1, 2, 0) 
             ax[1].imshow(transformed_image) 
             ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
             ax[1].axis("off")

             fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
   
   
def show_data(data):
   labels_map = data.classes

   figure = plt.figure(figsize=(8, 8))
   cols, rows = 3, 3
   for i in range(1, cols * rows + 1):
       sample_idx = torch.randint(len(data), size=(1,)).item()
       img, label = data[sample_idx]
       img = img.squeeze()
       figure.add_subplot(rows, cols, i)
       plt.title(labels_map[label], c="g")
       plt.imshow(img.permute(1, 2, 0), cmap="gray")
       plt.axis(False)
   plt.show()
    
    
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)

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

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

   
def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    # Put model in eval mode
    model.eval() 
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            t_loss = loss_fn(test_pred_logits, y)
            test_loss += t_loss()
            
            # Calculate and accumulate accuracy
            test_pred_labels = torch.argmax(test_pred_logits, dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_logits)
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acctest_acc += accuracy_fn(y_true=y,
                              y_pred=test_pred.argmax(dim=1))

 
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
      
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
         results["train_loss"].append(train_loss)
         results["train_acc"].append(train_acc)
         results["test_loss"].append(test_loss)
         results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results



# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)



def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();   
   
 

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


#  find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
