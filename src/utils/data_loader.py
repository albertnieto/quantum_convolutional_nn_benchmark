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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np

def load_mnist_data(batch_size=4, img_size=8, limit=None, output="dl"):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Limit dataset size if the limit parameter is provided
    if limit is not None:
        train_dataset = Subset(train_dataset, range(limit))
        test_dataset = Subset(test_dataset, range(limit))

    if output == "dl":
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    elif output == "np":
        # Collect data into NumPy arrays
        X_train = []
        y_train = []
        for data, target in DataLoader(train_dataset, batch_size=len(train_dataset)):
            X_train.append(data.numpy())
            y_train.append(target.numpy())

        X_test = []
        y_test = []
        for data, target in DataLoader(test_dataset, batch_size=len(test_dataset)):
            X_test.append(data.numpy())
            y_test.append(target.numpy())

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)

        return X_train, y_train, X_test, y_test
    else:
        raise ValueError(f"Unsupported format: {output}. Use 'dl' or 'np'.")
