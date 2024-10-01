# tests/test_templates.py

import pytest
import torch
import pennylane as qml
import numpy as np

# Import templates from convolution
from src.circuits.convolution.templates import (
    strongly_entangling_circuit,
    basic_entangler_circuit,
    simplified_two_design_circuit,
    random_layers_circuit,
)

# Import templates from embedding
from src.circuits.embedding.templates import (
    angle_embedding,
    amplitude_embedding,
    displacement_embedding,
    squeezing_embedding,
    qaoa_embedding,
)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of wires for testing
num_wires = 4
wires = range(num_wires)

# Create a quantum device
dev_qubit = qml.device("default.qubit", wires=num_wires)

# Create a CV device for continuous-variable embeddings
try:
    # Use default.gaussian device
    dev_cv = qml.device("default.gaussian", wires=num_wires)
except ValueError:
    dev_cv = None  # Handle the case where the CV device is not available

# Helper function to execute a QNode and check for differentiability
def execute_qnode(qnode, inputs, params):
    output = qnode(inputs, params)
    assert output is not None
    # Convert output to a torch tensor
    if isinstance(output, (list, tuple)):
        output = torch.tensor(output, device=device, dtype=torch.float64, requires_grad=True)
    elif isinstance(output, np.ndarray):
        output = torch.from_numpy(output).to(device).requires_grad_()
    # Check differentiability
    try:
        loss = torch.sum(output)
        loss.backward()
        assert inputs.grad is not None
        for param in params.values():
            if isinstance(param, torch.Tensor) and param.requires_grad:
                assert param.grad is not None
    except RuntimeError as e:
        pytest.fail(f"Backward pass failed: {e}")

# Test functions for convolution templates
def test_strongly_entangling_circuit():
    @qml.qnode(dev_qubit, interface="torch", diff_method="backprop")
    def circuit(inputs, params):
        angle_embedding(inputs, wires, params={})
        strongly_entangling_circuit(wires, params)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    inputs = torch.randn(num_wires, device=device, requires_grad=True)
    num_layers = 2
    weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_wires)
    weights = torch.randn(weights_shape, device=device, requires_grad=True)
    params = {'weights': weights}

    execute_qnode(circuit, inputs, params)

def test_basic_entangler_circuit():
    @qml.qnode(dev_qubit, interface="torch", diff_method="backprop")
    def circuit(inputs, params):
        angle_embedding(inputs, wires, params={})
        basic_entangler_circuit(wires, params)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    inputs = torch.randn(num_wires, device=device, requires_grad=True)
    num_layers = 2
    weights = torch.randn((num_layers, num_wires), device=device, requires_grad=True)
    params = {'weights': weights, 'rotation': qml.RX}

    execute_qnode(circuit, inputs, params)

def test_simplified_two_design_circuit():
    @qml.qnode(dev_qubit, interface="torch", diff_method="backprop")
    def circuit(inputs, params):
        angle_embedding(inputs, wires, params={})
        simplified_two_design_circuit(wires, params)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    inputs = torch.randn(num_wires, device=device, requires_grad=True)
    initial_layer_weights = torch.randn(num_wires, device=device, requires_grad=True)
    num_layers = 2
    weights = torch.randn((num_layers, num_wires - 1, 2), device=device, requires_grad=True)
    params = {'initial_layer_weights': initial_layer_weights, 'weights': weights}

    execute_qnode(circuit, inputs, params)

def test_random_layers_circuit():
    @qml.qnode(dev_qubit, interface="torch", diff_method="backprop")
    def circuit(inputs, params):
        angle_embedding(inputs, wires, params={})
        random_layers_circuit(wires, params)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    inputs = torch.randn(num_wires, device=device, requires_grad=True)
    num_layers = 2
    num_rots = num_wires
    weights = torch.randn((num_layers, num_rots), device=device, requires_grad=True)
    params = {
        'weights': weights,
        'ratio_imprim': 0.3,
        'imprimitive': qml.CNOT,
        'rotations': [qml.RX, qml.RY, qml.RZ],
        'seed': 42,
    }

    execute_qnode(circuit, inputs, params)

# Test functions for embedding templates
def test_angle_embedding():
    @qml.qnode(dev_qubit, interface="torch", diff_method="backprop")
    def circuit(inputs, params):
        angle_embedding(inputs, wires, params)
        # Use a simple variational circuit
        qml.StronglyEntanglingLayers(params['weights'], wires=wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    inputs = torch.randn(num_wires, device=device, requires_grad=True)
    num_layers = 2
    weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_wires)
    weights = torch.randn(weights_shape, device=device, requires_grad=True)
    params = {'weights': weights, 'rotation': 'X'}

    execute_qnode(circuit, inputs, params)

def test_amplitude_embedding():
    @qml.qnode(dev_qubit, interface="torch", diff_method="backprop")
    def circuit(inputs, params):
        amplitude_embedding(inputs, wires, params)
        # Use a simple variational circuit
        qml.StronglyEntanglingLayers(params['weights'], wires=wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    # The number of features must be 2^n for amplitude embedding
    inputs = torch.randn(2 ** num_wires, device=device, requires_grad=True)
    num_layers = 2
    weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=num_wires)
    weights = torch.randn(weights_shape, device=device, requires_grad=True)
    params = {'weights': weights, 'normalize': True, 'pad_with': 0.0}

    execute_qnode(circuit, inputs, params)

@pytest.mark.skipif(dev_cv is None, reason="default.gaussian device is not available")
def test_displacement_embedding():
    @qml.qnode(dev_cv, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, params):
        displacement_embedding(inputs, wires, params)
        return qml.expval(qml.X(wires[0]))

    inputs = torch.randn(num_wires, device=device, requires_grad=True)
    params = {'method': 'amplitude', 'c': 0.1}

    execute_qnode(circuit, inputs, params)

@pytest.mark.skipif(dev_cv is None, reason="default.gaussian device is not available")
def test_squeezing_embedding():
    @qml.qnode(dev_cv, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, params):
        squeezing_embedding(inputs, wires, params)
        return qml.expval(qml.X(wires[0]))

    inputs = torch.randn(num_wires, device=device, requires_grad=True)
    params = {'method': 'phase', 'c': 0.1}

    execute_qnode(circuit, inputs, params)

def test_qaoa_embedding():
    @qml.qnode(dev_qubit, interface="torch", diff_method="backprop")
    def circuit(inputs, params):
        qaoa_embedding(inputs, wires, params)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    inputs = torch.randn(num_wires, device=device, requires_grad=True)
    n_layers = 2
    weights_shape = qml.QAOAEmbedding.shape(n_layers=n_layers, n_wires=num_wires)
    weights = torch.randn(weights_shape, device=device, requires_grad=True)
    params = {'weights': weights, 'local_field': 'Y', 'n_layers': n_layers}

    execute_qnode(circuit, inputs, params)

# Run the tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__])
