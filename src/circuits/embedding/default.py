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

def default_embedding(inputs, wires, params):
    rotation = params.get('rotation', 'Z')
    qml.AngleEmbedding(inputs, wires=wires, rotation=rotation)

def custom_embedding(inputs, wires, params):
    normalize = params.get('normalize', True)
    pad_with = params.get('pad_with', 0.0)
    qml.AmplitudeEmbedding(inputs, wires=wires, normalize=normalize, pad_with=pad_with)