# Copyright 2024 CTIC (Technological Center for Information and Communication).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

src_path = os.path.abspath(os.path.join('..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from layers.quanvolution import QuanvLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuanvolutionalNet(nn.Module):
    def __init__(self, qkernel_shape=2, classical_kernel_shape=3, embedding=None,
                 circuit=None, measurement=None, params=None, qdevice_kwargs=None,
                 n_classes=10, batch_size=32, epochs=10, learning_rate=None):
        super(QuanvolutionalNet, self).__init__()
        self.qkernel_shape = qkernel_shape
        self.device = device
        self.params = params or {}
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.embedding = embedding
        self.circuit = circuit
        self.measurement = measurement
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.quanv = QuanvLayer(
            qkernel_shape=qkernel_shape,
            embedding=embedding,
            circuit=circuit,
            measurement=measurement,
            params=self.params,
            qdevice_kwargs=self.qdevice_kwargs
        ).to(self.device)
        in_channels = self.qkernel_shape ** 2

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=classical_kernel_shape).to(self.device)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, padding=1).to(self.device)
        self.fc1 = None
        self.fc2 = nn.Linear(128, n_classes).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.quanv(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(self.device)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

    def fit(self, X_train=None, y_train=None, train_loader=None,
            criterion=nn.CrossEntropyLoss(), optimizer=None, epochs=None, batch_size=None):

        if epochs is not None:
            self.epochs = epochs

        if batch_size is not None:
            self.batch_size = batch_size

        if train_loader is None and X_train is not None and y_train is not None:
            if isinstance(X_train, np.ndarray):
                X_train = torch.tensor(X_train, dtype=torch.float32)
            if isinstance(y_train, np.ndarray):
                y_train = torch.tensor(y_train, dtype=torch.long)

            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)

            dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.train()
        self.to(self.device)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss:.4f}")

    def predict(self, X=None, batch_size=32):
        self.eval()
        self.to(self.device)

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)

        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.append(predicted.cpu())

        all_predictions = torch.cat(all_predictions)

        return all_predictions

    def __name__(self):
        return self.__class__.__name__
