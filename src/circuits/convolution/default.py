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
import pennylane as qml
from pennylane import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def default_circuit(wires, params):
    num_layers = params.get('num_layers', 1)
    weights = params.get('weights', torch.randn(num_layers, len(wires), device=device))

    qml.templates.RandomLayers(weights, wires=wires)

def full_entanglement_circuit(wires, params):
    n_qubits = len(wires)
    num_layers = params.get('num_layers', 1)
    weights = params.get('weights', torch.randn(num_layers, n_qubits, 3, device=device) % np.pi)

    qml.templates.StronglyEntanglingLayers(weights, wires=wires)