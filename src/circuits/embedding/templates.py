# src/circuits/embedding/templates.py

import torch
import pennylane as qml

def angle_embedding(inputs, wires, params):
    """
    An embedding using the AngleEmbedding template from PennyLane.
    """
    rotation = params.get('rotation', 'X')
    qml.templates.AngleEmbedding(features=inputs, wires=wires, rotation=rotation)

def amplitude_embedding(inputs, wires, params):
    """
    An embedding using the AmplitudeEmbedding template from PennyLane.
    """
    normalize = params.get('normalize', True)
    pad = params.get('pad_with', 0.0)
    qml.templates.AmplitudeEmbedding(features=inputs, wires=wires, normalize=normalize, pad_with=pad)

def displacement_embedding(inputs, wires, params):
    """
    An embedding using the DisplacementEmbedding template from PennyLane.

    Note: This embedding is designed for continuous-variable (CV) devices.
    """
    method = params.get('method', 'amplitude')
    c = params.get('c', 0.1)
    qml.templates.DisplacementEmbedding(features=inputs, wires=wires, method=method, c=c)

def squeezing_embedding(inputs, wires, params):
    """
    An embedding using the SqueezingEmbedding template from PennyLane.

    Note: This embedding is designed for continuous-variable (CV) devices.
    """
    method = params.get('method', 'amplitude')
    c = params.get('c', 0.1)
    qml.templates.SqueezingEmbedding(features=inputs, wires=wires, method=method, c=c)

def qaoa_embedding(inputs, wires, params):
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
