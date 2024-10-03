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

def amplitude_embedding(inputs, wires, params):
    qml.AmplitudeEmbedding(inputs, wires = wires, pad_with = 0.0, normalize = True, id = None)

def angle_embedding(inputs, wires, params):
    qml.AngleEmbedding(inputs, wires = wires)

def QAOA_embedding(inputs, wires, params): #PROVISIONAL
    qkernel_shape = params.get('qkernel_shape')

    if qkernel_shape == 2:
        n = 8
    elif qkernel_shape == 3:
        n = 18
    qml.QAOAEmbedding(inputs, torch.rand(2,n)*np.pi, wires = wires)

def displacement_embedding(inputs, wires, params):
    method = params.get('method')
    qml.DisplacementEmbedding(inputs, wires = wires, method = method)

def squeezing_embedding(inputs, wires, params):
    method = params.get('method')
    qml.SqueezingEmbedding(inputs, wires = wires, method = method)