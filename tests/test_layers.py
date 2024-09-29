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

import pytest
import torch
from src.layers import QuanvLayer
from src.circuits.convolution import custom_circuit
from src.circuits.embedding import custom_embedding
from src.circuits.measurement import custom_measurement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_quanv_layer():
    layer = QuanvLayer(
        qkernel_shape=2,
        embedding=custom_embedding,
        circuit=custom_circuit,
        measurement=custom_measurement,
        params={'circuit': {'weights': torch.randn(2, 4, 3, device=device)}}
    )
    x = torch.randn(1, 1, 8, 8)
    try:
        output = layer(x)
        assert output.shape[0] == 1
    except Exception as e:
        pytest.fail(f"QuanvLayer raised an exception: {e}")
