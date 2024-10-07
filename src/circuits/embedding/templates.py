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
    """
    An embedding using the AmplitudeEmbedding template from PennyLane.
    """
    normalize = params.get('normalize', True)
    pad_with = params.get('pad_with', 0.0)
    qml.AmplitudeEmbedding(inputs, wires = wires, pad_with = pad_with, normalize = normalize)

def angle_embedding(inputs, wires, params):
    """
    An embedding using the AngleEmbedding template from PennyLane.
    """
    rotation = params.get('rotation', 'X')
    qml.AngleEmbedding(inputs, wires = wires, rotation=rotation)

def QAOA_embedding(inputs, wires, params):
    """
    An embedding using the QAOAEmbedding template from PennyLane.

    This embedding is differentiable and suitable for qubit-based devices.
    """
    weights = params.get('weights')
    local_field = params.get('local_field', 'Y')
    n_layers = params.get('n_layers', 1)
    if weights is None:
        weights_shape = qml.templates.QAOAEmbedding.shape(n_layers=n_layers, n_wires=len(wires))
        weights = torch.randn(weights_shape, requires_grad=True)
    qml.templates.QAOAEmbedding(features=inputs, weights=weights, wires=wires, local_field=local_field)

def displacement_embedding(inputs, wires, params):
    """
    An embedding using the DisplacementEmbedding template from PennyLane.

    Note: This embedding is designed for continuous-variable (CV) devices.
    """
    c = params.get('c', 0.1)
    method = params.get('method', 'amplitude')
    qml.DisplacementEmbedding(inputs, wires = wires, method = method)

def squeezing_embedding(inputs, wires, params):
    """
    An embedding using the SqueezingEmbedding template from PennyLane.

    Note: This embedding is designed for continuous-variable (CV) devices.
    """
    c = params.get('c', 0.1)
    method = params.get('method', 'amplitude')
    qml.SqueezingEmbedding(inputs, wires = wires, method = method, c = c)