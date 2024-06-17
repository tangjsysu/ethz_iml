# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
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
from efficientnet_pytorch import EfficientNet

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    # The required pre-processing depends on the pre-trained model you choose
    # below.
    # See https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(root="../data/task3/dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't
    # run out of memory (VRAM if on GPU, RAM if on CPU)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)

    # Load the EfficientNet
    embeddings_model = models.mobilenet_v3_large()

    # Remove the last layer (classifier)
    embeddings_model._fc = nn.Identity()
    print(embeddings_model)

    embeddings_model.to(device)
    embedding_size = 2560  # Dummy variable, replace with the actual embedding size once you
    # pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the
    # model to access the embeddings the model generates.
    embeddings_model.to(device)

    def extract_embeddings(model, dataloader):
        embeddings = []
        model.eval()
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc='Extracting embeddings'):
                images = images.to(device)
                outputs = model(images)
                embeddings.append(outputs.cpu().numpy())
        embeddings = np.concatenate(embeddings)
        return embeddings

    embeddings = extract_embeddings(embeddings_model, train_loader)
    np.save('../data/task3/dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="../data/task3/dataset/",
                                         transform=None)
    filenames = [s[0].split('\\')[-1].replace('.jpg', '') for s in train_dataset.samples]


    embeddings = np.load('D:/anaconda/envs/NLP/iml/data/task3/dataset/embeddings.npy')
    # TODO: Normalize the embeddings
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings_normalized[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
   # print(file_to_embedding)
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    print(X.shape)
    y = np.hstack(y)
    return X, y


# Hint: adjust batch_size and num_workers to your PC configuration, so that you
# don't run out of memory (VRAM if on GPU, RAM if on CPU)
def create_loader_from_np(X, y=None, train=True, batch_size=32, shuffle=True):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        # Attention: If you get type errors you can modify the type of the
        # labels here
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float),
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle)
    return loader


# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(7680),
            nn.Linear(7680, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
        )


    def forward(self, x):
        out = self.linear_relu_stack(x)
        out = F.sigmoid(out)
        return out


def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    # Define loss function and optimizer
    criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    n_epochs = 5
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for i, (X, y) in tqdm(enumerate(train_loader), desc="DataLoader"):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            predictions = torch.round(outputs)  # Round the output to 0/1
            correct_predictions += (predictions == y.unsqueeze(1)).sum().item()
            total_predictions += y.size(0)

        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}")

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
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("../data/task3/results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = '../data/task3/train_triplets.txt'
    TEST_TRIPLETS = '../data/task3/test_triplets.txt'

    # generate embedding for each image in the dataset
    if (os.path.exists('../data/task3/dataset/embeddings.npy') == False):
        generate_embeddings()

   # load the training data
    X, y = get_data(TRAIN_TRIPLETS)
    # Create data loaders for the training data

    train_loader = create_loader_from_np(X, y, train=True, batch_size=32)
    # delete the loaded training data to save memory, as the data loader copies
    del X
    del y

    # repeat for testing data
    X_test, y_test = get_data(TEST_TRIPLETS, train=False)
    test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False)
    del X_test
    del y_test

    # define a model and train it
    model = train_model(train_loader)

    # test the model on the test data
    tst_model(model, test_loader)
    print("Results saved to results.txt")
