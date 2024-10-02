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

def custom_iqp_embedding(inputs, wires, params):
    n_qubits = len(wires)
    n_repeats = params.get("n_repeats")
    pattern = params.get("pattern")


    # Repeat the pattern n_repeats times
    for _ in range(n_repeats):
        
        for idx, wire in enumerate(wires):

            # Apply Hadamard gates to all wires
            qml.Hadamard(wires = wire)

            # Apply RZ rotations with inputs as parameters
            qml.RZ(inputs[:, idx], wires=wire)

        # Apply entangling gates ZZ
        if pattern is None:
            # Default pattern: all combinations of qubits
            cont = n_qubits - 1
            a = 0
            for i in range(n_qubits - 1):
                for j in range(cont):
                    wire1 = wires[a]
                    wire2 = wires[j + 1 + a]
                    phi = inputs[:, a] * inputs[:, j + a + 1]
                    qml.IsingZZ(phi, wires = [wire1, wire2])
                cont -= 1
                a += 1
                
        else:
            # Custom pattern
            for pair in pattern:
                wire1 = wires[pair[0]]
                wire2 = wires[pair[1]]
                phi = inputs[:, pair[0]] * inputs[:, pair[1]]
                qml.ZZ(phi, wires=[wire1, wire2])
                #qml.RZ(np.random.rand()*np.pi, wires=wire2)