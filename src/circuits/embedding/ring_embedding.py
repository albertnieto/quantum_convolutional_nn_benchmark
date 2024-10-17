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

def ring_embedding(inputs, wires, params):

    n_repeats = params.get("n_repeats", 1)
    n_qubits = len(wires)
    
    # Repeat the pattern n_repeats times
    for _ in range(n_repeats):
        
        for idx, wire in enumerate(wires):

            # Apply Hadamard gates to all wires
            qml.Hadamard(wires = wire)

            # Apply RZ rotations with inputs as parameters
            qml.RY(inputs[:, idx], wires=wire)


        for i in range(n_qubits):
            if i < n_qubits - 1:   
                qml.CNOT(wires = [i, i + 1])
            else:
                qml.CNOT(wires = [i, 0])

        for idx, wire in enumerate(wires):

            # Apply RZ rotations with inputs as parameters
            qml.RY(inputs[:, idx], wires=wire)

        for i in range(n_qubits):
            if i < n_qubits - 1:   
                qml.CNOT(wires = [i + 1, i])
            else:
                qml.CNOT(wires = [0, i])