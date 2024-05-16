import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class LiDARDataset(Dataset):
    def __init__(self, num_samples=100, max_points=50):
        self.data = []
        for _ in range(num_samples):
            num_points = np.random.randint(10, max_points)
            points = torch.rand(num_points, 4)  # Generate random x, y, z, and intensity values
            self.data.append(points)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=0)

class LiDARTransformer(nn.Module):
    def __init__(self):
        super(LiDARTransformer, self).__init__()
        self.embedding = nn.Linear(4, 768)  # Embedding layer to transform 4 features into 768-dim vector
        self.transformer = BertModel(BertConfig())
        self.classifier = nn.Linear(768, 1)  # Output a score for each point

    def forward(self, x):
        attention_mask = (x.sum(dim=-1) != 0).float()  # Compute attention mask
        x = self.embedding(x)  # Embed 4D point data into higher dimensional space
        transformer_output = self.transformer(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        scores = self.classifier(transformer_output).squeeze(-1)
        return scores

def minimize_intensities(scores, data):
    probabilities = torch.sigmoid(scores)
    weighted_intensities = probabilities * data[:, :, 3]
    return torch.sum(weighted_intensities)

def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        scores = model(data)
        loss = minimize_intensities(scores, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Average Loss: {total_loss / len(data_loader)}")

dataset = LiDARDataset()
data_loader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.collate_fn)
model = LiDARTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train(model, data_loader, optimizer)
