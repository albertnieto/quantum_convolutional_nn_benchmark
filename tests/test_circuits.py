# tests/test_circuits.py

import pytest
import torch
from src.circuits.convolution.default import default_circuit
from src.circuits.embedding.default import default_embedding
from src.circuits.measurement.default import default_measurement
import pennylane as qml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_default_circuit():
    """
    Test the default_circuit function to ensure it runs without errors.
    """
    wires = [0, 1, 2, 3]
    params = {'num_layers': 1, 'weights': torch.randn(1, 4, device=device)}
    inputs = torch.randn(4, device=device)
    try:
        default_circuit(inputs, wires, params)
    except Exception as e:
        pytest.fail(f"default_circuit raised an exception: {e}")
