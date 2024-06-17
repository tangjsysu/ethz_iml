# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.preprocessing import normalize
from tqdm import tqdm
import torch.optim as optim
from PIL import Image

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TripletDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data = np.loadtxt(txt_file, dtype=str)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_names = self.data[idx]
        images = [Image.open('../data/task3/dataset/food/' + image_name + '.jpg') for image_name in image_names]

        if self.transform:
            images = [self.transform(image) for image in images]

        return images

def create_dataloader(txt_file, batch_size=32, shuffle=True, transform=None):
    dataset = TripletDataset(txt_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# 使用示例

class TripletLoss(nn.Module):
    """
    Triplet loss function.
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # Euclidean distance
        distance_negative = (anchor - negative).pow(2).sum(1)  # Euclidean distance
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """

    def __init__(self, pretrained_model):
        """
        The constructor of the model.
        """
        super().__init__()
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128)
        )

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.linear_relu_stack(self.pretrained_model(anchor))
        positive_embedding = self.linear_relu_stack(self.pretrained_model(positive))
        negative_embedding = self.linear_relu_stack(self.pretrained_model(negative))
        return anchor_embedding, positive_embedding, negative_embedding


def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """
    pretrained_model = models.resnet101(pretrained=True)
    del pretrained_model.fc
    model = Net(pretrained_model)
    model.train()
    model.to(device)
    # Define loss function and optimizer
    criterion = TripletLoss()# Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    n_epochs = 13
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (batch_images) in tqdm(enumerate(train_loader), desc="DataLoader"):
            anchor, positive, negative = batch_images
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    return model

def tst_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data

    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad():  # We don't need to compute gradients for testing
        for batch_images in loader:
            anchor, positive, negative = batch_images
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            # Compute the distances between the embeddings
            distance_positive = (anchor_embedding - positive_embedding).pow(2).sum(1)  # Euclidean distance
            distance_negative = (anchor_embedding - negative_embedding).pow(2).sum(1)  # Euclidean distance
            # Make the predictions based on the distances
            predicted = (distance_positive < distance_negative).cpu().numpy()
            predictions.append(predicted)
        predictions = np.hstack(predictions)
    np.savetxt("../data/task3/results.txt", predictions, fmt='%i')

# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = '../data/task3/train_triplets.txt'
    TEST_TRIPLETS = '../data/task3/test_triplets.txt'
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataloader = create_dataloader(TRAIN_TRIPLETS, batch_size=32, shuffle=True, transform=transform)
    test_dataloader = create_dataloader(TEST_TRIPLETS, batch_size=128, shuffle=False, transform=transform_test)
    model = train_model(train_dataloader)
    tst_model(model, test_dataloader)
    #model = train_model(train_dataloader)
   # tst_model(model, test_dataloader)
    print("Results saved to results.txt")
    # Create data loaders for the training data
