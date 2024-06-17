import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

class CommentDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.data.iloc[idx]["title"]
        sentence = self.data.iloc[idx]["sentence"]
        score = self.data.iloc[idx]["score"]

        inputs_title = self.tokenizer(title, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        inputs_sentence = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length",
                                         max_length=256)

        # Squeeze the extra dimension and convert to tensor
        inputs_title = {'input_ids': inputs_title['input_ids'].squeeze(0),
                        'attention_mask': inputs_title['attention_mask'].squeeze(0)}
        inputs_sentence = {'input_ids': inputs_sentence['input_ids'].squeeze(0),
                           'attention_mask': inputs_sentence['attention_mask'].squeeze(0)}

        score = torch.tensor(score, dtype=torch.float)

        return inputs_title, inputs_sentence, score
class TestDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.data.iloc[idx]["title"]
        sentence = self.data.iloc[idx]["sentence"]

        inputs_title = self.tokenizer(title, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs_sentence = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length",
                                         max_length=128)

        inputs_title = {'input_ids': inputs_title['input_ids'].squeeze(0),
                        'attention_mask': inputs_title['attention_mask'].squeeze(0)}
        inputs_sentence = {'input_ids': inputs_sentence['input_ids'].squeeze(0),
                           'attention_mask': inputs_sentence['attention_mask'].squeeze(0)}

        return inputs_title, inputs_sentence






class BERTEmbeddingModel(nn.Module):
    def __init__(self):
        super(BERTEmbeddingModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1))

    def forward(self, input_title, input_sentence):
        outputs_title = self.bert(**input_title)
        outputs_sentence = self.bert(**input_sentence)
        last_hidden_state_title = outputs_title.last_hidden_state
        last_hidden_state_sentence = outputs_sentence.last_hidden_state
        # add the embeddings of the title and sentence
        outputs = last_hidden_state_title + last_hidden_state_sentence
        pooled_output = outputs[:, 0, :]
        scores = self.linear_relu_stack(pooled_output)
        return scores




def train(train_loader, val_loader):
    model = BERTEmbeddingModel().to(device)

    # train the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10
    best_val_loss = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (input_title, input_sentence, labels) in enumerate(
                tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
            input_title = {key: value.to(device) for key, value in input_title.items()}
            input_sentence = {key: value.to(device) for key, value in input_sentence.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_title, input_sentence)
            loss = nn.MSELoss()(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # print loss every batch
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}")

        # evaluate the model on the validation set
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for input_title, input_sentence, labels in val_loader:
                input_title = {key: value.to(device) for key, value in input_title.items()}
                input_sentence = {key: value.to(device) for key, value in input_sentence.items()}
                labels = labels.to(device)

                outputs = model(input_title, input_sentence)
                val_loss = nn.MSELoss()(outputs.squeeze(), labels)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        if best_val_loss == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "../data/task4/product_rating_model.pth")

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


def predict(data):
    model = BERTEmbeddingModel().to(device)
    model.load_state_dict(torch.load("../data/task4/product_rating_model.pth"))
    model.eval()
    predictions = []
    with torch.no_grad():
        for input_title, input_sentence in tqdm(data, desc="Predicting"):
            input_title = {key: value.to(device) for key, value in input_title.items()}
            input_sentence = {key: value.to(device) for key, value in input_sentence.items()}

            outputs = model(input_title, input_sentence)
            predictions.extend(outputs.squeeze().cpu().tolist())
    print(predictions)
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
    np.savetxt("../data/task4/results.txt", predictions)

if __name__ == '__main__':
    train_data = pd.read_csv("../data/task4/train.csv")  # load data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # initialize tokenizer

    # split data into training and validation sets
    train_df, val_df = train_test_split(train_data, test_size=0.1, random_state=42)

    # create dataloaders
    train_dataset = CommentDataset(train_df, tokenizer)
    val_dataset = CommentDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_data = pd.read_csv("../data/task4/test_no_score.csv")
    test_dataset = TestDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    #train(train_loader, val_loader)
    predict(test_loader)