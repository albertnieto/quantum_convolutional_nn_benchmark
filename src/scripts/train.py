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
import torch.optim as optim
import torch.nn as nn
from src.models.quanvolutional_net import QuanvolutionalNet
from src.utils.data_loader import load_mnist_data
from src.circuits.convolution import custom_circuit
from src.circuits.embedding import custom_embedding
from src.circuits.measurement import custom_measurement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(batch_size=4, output='np', limit=250):
    return load_mnist_data(batch_size=batch_size, output=output, limit=limit)

def initialize_model(qkernel_shape, classical_kernel_shape, n_classes, num_layers=2):
    num_wires = qkernel_shape ** 2
    weights = torch.randn(num_layers, num_wires, 3, device=device)

    params = {
        'circuit': {'weights': weights}
    }

    model = QuanvolutionalNet(
        qkernel_shape=qkernel_shape,
        classical_kernel_shape=classical_kernel_shape,
        embedding=custom_embedding,
        circuit=custom_circuit,
        measurement=custom_measurement,
        params=params,
        n_classes=n_classes
    )

    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.fit(
        X_train=torch.tensor(X_train, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.long),
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size
    )

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(torch.tensor(X_test, dtype=torch.float32))
    accuracy = (predictions == torch.tensor(y_test)).float().mean()
    return accuracy.item()

def main(batch_size=4, limit=250, epochs=10, batch_size_train=32, suppress_print=False):
    # Load data
    X_train, y_train, X_test, y_test = load_data(batch_size=batch_size, output='np', limit=limit)

    # Define model parameters
    qkernel_shape = 2
    classical_kernel_shape = 3
    n_classes = 10

    # Initialize model
    model = initialize_model(qkernel_shape, classical_kernel_shape, n_classes)

    # Train model
    train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)

    if not suppress_print:
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy


if __name__ == "__main__":
    main()
