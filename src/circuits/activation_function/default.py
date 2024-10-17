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

def default_activation(wires, params):
    weights = params.get("weights", torch.rand(len(wires), device = device) % np.pi)
    qml.Hadamard(wires = wires)
    for i in range(len(wires)):
        qml.RY(weights[i], wires = i)


def custom_activation(inputs, wires, params):
    normalize = params.get('normalize', True)
    pad_with = params.get('pad_with', 0.0)
    qml.AmplitudeEmbedding(inputs, wires=wires, normalize=normalize, pad_with=pad_with)