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
from layers.quanvolution import QuanvLayer
from utils.plot_cm import confusion_matrix_plot
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuanvolutionalNet(nn.Module):
    def __init__(self, qkernel_shape=2, classical_kernel_shape=3, embedding=None,
                 circuit=None, measurement=None, params=None, qdevice_kwargs=None,
                 n_classes=10, batch_size=32, epochs=10, learning_rate=1e-3,
                 criterion=nn.CrossEntropyLoss(), optimizer_class=optim.Adam, use_quantum=True, plot=True, data=None, optimizer_kwargs=None):
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
        self.use_quantum = use_quantum
        self.confusion_matrix = None
        self.plot = plot
        self.data = data

        if data == 'MNIST':
            labels = ['0', '1','2','3', '4','5', '6','7', '8','9']
            
        elif data == 'Eurosat':
            labels = ['Annual\nCrop', 'Forest',
                  'Herbaceous\nVegetation',
                  'Highway', 'Industrial',
                  'Pasture', 'Permanent\nCrop',
                  'Residential', 'River',
                  'SeaLake']
            
        allowed_class_idx = None
        self.labels = labels if allowed_class_idx is None else [labels[i] for i in allowed_class_idx if i < len(labels)]

        if use_quantum:
            self.quanv = QuanvLayer(
                qkernel_shape=qkernel_shape,
                embedding=embedding,
                circuit=circuit,
                measurement=measurement,
                params=self.params,
                qdevice_kwargs=self.qdevice_kwargs
            ).to(self.device)
            
            if data == 'MNIST':
                in_channels = (qkernel_shape**2)
                
            elif data == 'Eurosat':
                in_channels = 3*(qkernel_shape**2)
            
            kernel_size = 7 if qkernel_shape==2 else 6
            kernel_size2 = 2 
                
        else:
            if data == 'MNIST':
                kernel_size2 = 2
                self.conv1_classical = nn.Conv2d(1, qkernel_shape**5, kernel_size=8).to(self.device)
            elif data == 'Eurosat':
                kernel_size2 = 1
                self.conv1_classical = nn.Conv2d(3, qkernel_shape**5, kernel_size=8).to(self.device)
            in_channels = qkernel_shape**5
            kernel_size = 1   

        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=kernel_size).to(self.device)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=kernel_size2, padding=1).to(self.device)
        self.fc1 = None
        self.fc2 = nn.Linear(128, n_classes).to(self.device)

        # Initialize criterion and optimizer
        self.criterion = criterion
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_class(self.parameters(), lr=self.learning_rate, **optimizer_kwargs)

        # Initialize tracking variables
        self.train_losses = []
        self.train_accuracies = []

    def forward(self, x):
        x = x.to(self.device)
        if self.use_quantum:
            x = self.quanv(x)
        else:
            x = self.conv1_classical(x)
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
            epochs=None, batch_size=None):
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

        self.train_losses = []
        self.train_accuracies = []
        all_labels = []
        all_preds = []

        self.train()
        self.to(self.device)

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

            average_loss = running_loss / total_samples
            accuracy = correct_predictions / total_samples
            self.train_losses.append(average_loss)
            self.train_accuracies.append(accuracy)
            self.confusion_matrix = confusion_matrix(all_labels, all_preds)

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
            #print(f'    Confusion Matrix:\n{self.confusion_matrix}')

        if self.plot:
            confusion_matrix_plot(self.confusion_matrix, self.labels)

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
