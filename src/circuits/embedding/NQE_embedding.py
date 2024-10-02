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
import numpy.pi as pi

def NQE_embedding(inputs, wires, params):

    n_qubits = len(wires)
    n_repeats = params.get('n_repeats')

    for _ in range(n_repeats):

        for idx, wire in enumerate(wires):

            qml.Hadamard(wires = wire)
            qml.RZ(inputs[:, idx], wires = wire)

        cont = n_qubits - 1
        a = 0
        for _ in range(n_qubits - 1):
            for j in range(cont):
                wire1 = wires[a]
                wire2 = wires[j + 1 + a]
                phi = (pi - inputs[:, a]) * (pi - inputs[:, j + a + 1])

                qml.CNOT(wires = [wire1, wire2])
                qml.RZ(phi, wires = wire2)
                qml.CNOT(wires = [wire1, wire2])
            cont -= 1
            a += 1