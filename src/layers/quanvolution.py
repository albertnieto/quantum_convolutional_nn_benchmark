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
import pennylane as qml

import sys
import os

from circuits.convolution import default_circuit
from circuits.embedding import default_embedding
from circuits.measurement import default_measurement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuanvLayer(nn.Module):
    def __init__(self, qkernel_shape, embedding=None, circuit=None, measurement=None, params=None, qdevice_kwargs=None):
        super(QuanvLayer, self).__init__()
        self.qkernel_shape = qkernel_shape
        self.embedding = embedding or default_embedding
        self.circuit = circuit or default_circuit
        self.measurement = measurement or default_measurement
        self.params = params or {}
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.torch_device = device
        self.qml_device = None
        self.qnode = None

    def quantum_circuit(self, inputs):
        wires = range(self.qkernel_shape ** 2)
        params = self.params

        # Embedding block
        self.embedding(inputs, wires, params.get('embedding', {}))

        # Circuit block
        self.circuit(wires, params.get('circuit', {}))

        # Measurement block
        return self.measurement(wires, params.get('measurement', {}))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.to(self.torch_device)
        patch_size = self.qkernel_shape ** 2

        if self.qnode is None:
            qml_device_name = self.qdevice_kwargs.pop('qml_device_name', 'default.qubit')
            self.qml_device = qml.device(
                qml_device_name, wires=patch_size, **self.qdevice_kwargs
            )
            self.qnode = qml.QNode(
                self.quantum_circuit,
                self.qml_device,
                interface='torch',
                diff_method='backprop'
            )

        # Extract patches
        patches = x.unfold(2, self.qkernel_shape, 1).unfold(3, self.qkernel_shape, 1)
        patches = patches.contiguous().view(-1, self.qkernel_shape ** 2)

        # Process patches
        outputs = self.qnode(patches)

        # Remove the torch.stack line
        outputs = torch.stack(outputs, dim=1)

        outputs = outputs.float()

        # Reshape outputs
        out_height = height - self.qkernel_shape + 1
        out_width = width - self.qkernel_shape + 1
        outputs = outputs.view(batch_size, -1, out_height, out_width)

        return outputs
