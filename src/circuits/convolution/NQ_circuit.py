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
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NQ_circuit(wires, params):
    
    n_qubits = len(wires)
    weights = params.get('weights')

    if weights is None:
        weights = torch.randn(3*n_qubits,  2, device=device)

    
    for wire in range(n_qubits-1):
        qml.RY(weights[wire, 0], wires = wire)
        qml.RZ(weights[wire, 1], wires = wire)

    qml.Barrier()
    
    for wire in range(n_qubits-2):
        qml.CNOT(wires = [wires[-wire-3], wires[-wire -2]])

    qml.Barrier()

    for wire in range(n_qubits-1):
        qml.RY(weights[wire + 4, 0], wires = wire)
        qml.RZ(weights[wire + 4, 1], wires = wire)

    qml.Barrier()
    
    for wire in range(n_qubits-2):
        qml.CNOT(wires = [wires[-wire-3], wires[-wire -2]])

    qml.Barrier()

    for wire in range(n_qubits-1):
        qml.RY(weights[wires + 8, 0], wires = wire)
        qml.RZ(weights[wires + 8, 1], wires = wire)