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
import pennylane as qml
from src.circuits.convolution import default_circuit, custom_circuit
from src.circuits.embedding import default_embedding, custom_embedding
from src.circuits.measurement import default_measurement, custom_measurement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_default_embedding():
    inputs = torch.randn(1, 4, device=device)
    wires = range(4)
    params = {}
    try:
        default_embedding(inputs, wires, params)
    except Exception as e:
        pytest.fail(f"default_embedding raised an exception: {e}")

def test_custom_embedding():
    inputs = torch.randn(1, 4, device=device)
    wires = range(4)
    params = {}
    try:
        custom_embedding(inputs, wires, params)
    except Exception as e:
        pytest.fail(f"custom_embedding raised an exception: {e}")

def test_default_circuit():
    wires = range(4)
    params = {}
    try:
        default_circuit(wires, params)
    except Exception as e:
        pytest.fail(f"default_circuit raised an exception: {e}")

def test_custom_circuit():
    wires = range(4)
    params = {'weights': torch.randn(2, 4, 3, device=device)}
    try:
        custom_circuit(wires, params)
    except Exception as e:
        pytest.fail(f"custom_circuit raised an exception: {e}")

def test_default_measurement():
    wires = range(4)
    params = {}
    try:
        results = default_measurement(wires, params)
        assert len(results) == len(wires)
    except Exception as e:
        pytest.fail(f"default_measurement raised an exception: {e}")

def test_custom_measurement():
    wires = range(4)
    params = {}
    try:
        results = custom_measurement(wires, params)
        assert len(results) == len(wires)
    except Exception as e:
        pytest.fail(f"custom_measurement raised an exception: {e}")
