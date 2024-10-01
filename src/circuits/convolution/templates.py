# src/circuits/convolution/templates.py

import torch
import pennylane as qml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def strongly_entangling_circuit(wires, params):
    """
    A convolutional circuit using the StronglyEntanglingLayers template from PennyLane.
    """
    weights = params.get('weights')
    if weights is None:
        num_layers = params.get('num_layers', 1)
        weights_shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=num_layers, n_wires=len(wires))
        weights = torch.randn(weights_shape, device=device, requires_grad=True)
    qml.templates.StronglyEntanglingLayers(weights, wires=wires)

def basic_entangler_circuit(wires, params):
    """
    A convolutional circuit using the BasicEntanglerLayers template from PennyLane.
    """
    weights = params.get('weights')
    rotation = params.get('rotation', qml.RX)
    if weights is None:
        num_layers = params.get('num_layers', 1)
        weights_shape = (num_layers, len(wires))
        weights = torch.randn(weights_shape, device=device, requires_grad=True)
    qml.templates.BasicEntanglerLayers(weights, wires=wires, rotation=rotation)

def efficient_su2_circuit(wires, params):
    """
    A convolutional circuit using the EfficientSU2 template from PennyLane.
    """
    weights = params.get('weights')
    entanglement = params.get('entanglement', 'full')
    if weights is None:
        num_layers = params.get('num_layers', 1)
        weights_shape = qml.templates.EfficientSU2.shape(n_wires=len(wires), n_layers=num_layers)
        weights = torch.randn(weights_shape, device=device, requires_grad=True)
    qml.templates.EfficientSU2(weights, wires=wires, entanglement=entanglement)

def simplified_two_design_circuit(wires, params):
    """
    A convolutional circuit using the SimplifiedTwoDesign template from PennyLane.
    """
    initial_layer_weights = params.get('initial_layer_weights')
    weights = params.get('weights')
    if initial_layer_weights is None:
        initial_layer_weights = torch.randn(len(wires), device=device, requires_grad=True)
    if weights is None:
        num_layers = params.get('num_layers', 1)
        weights_shape = (num_layers, len(wires) - 1, 2)
        weights = torch.randn(weights_shape, device=device, requires_grad=True)
    qml.templates.SimplifiedTwoDesign(initial_layer_weights, weights, wires=wires)

def random_layers_circuit(wires, params):
    """
    A convolutional circuit using the RandomLayers template from PennyLane.
    """
    weights = params.get('weights')
    if weights is None:
        num_layers = params.get('num_layers', 1)
        num_rots = params.get('num_rots', len(wires))
        weights_shape = (num_layers, num_rots)
        weights = torch.randn(weights_shape, device=device, requires_grad=True)
    ratio_imprim = params.get('ratio_imprim', 0.3)
    imprimitive = params.get('imprimitive', qml.CNOT)
    rotations = params.get('rotations', [qml.RX, qml.RY, qml.RZ])
    seed = params.get('seed', 42)
    qml.templates.RandomLayers(weights, wires=wires, ratio_imprim=ratio_imprim,
                               imprimitive=imprimitive, rotations=rotations, seed=seed)
