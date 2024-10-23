# src/grids/architecture/test.py

from src.layers.quanvolution import QuanvLayer
import torch.nn as nn
from src.circuits.convolution.default import default_circuit
from src.circuits.embedding.default import default_embedding
from src.circuits.measurement.default import default_measurement
import pennylane as qml  # Added import for qml

# Architecture 1: One QuanvLayer, Conv2d, another QuanvLayer
test_architecture_1 = [
    {
        'layer_class': QuanvLayer,
        'params': {
            'qkernel_shape': 2,
            'embedding': default_embedding,
            'circuit': default_circuit,
            'measurement': default_measurement,
            'params': {'observable': qml.PauliZ, 'rotation': 'X', 'num_layers': 1},
            # 'out_channels' is handled automatically
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': nn.Conv2d,
        'params': {
            'out_channels': 'auto',  # Set to 'auto'
            'kernel_size': 3,
            'padding': 1
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': QuanvLayer,
        'params': {
            'qkernel_shape': 3,
            'embedding': default_embedding,
            'circuit': default_circuit,
            'measurement': default_measurement,
            'params': {'observable': qml.PauliZ, 'rotation': 'X', 'num_layers': 1},
            # 'out_channels' is handled automatically
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {'layer_class': nn.Flatten, 'params': {}},
    {
        'layer_class': nn.Linear,
        'params': {'in_features': 'auto', 'out_features': 128}
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': nn.Linear,
        'params': {'in_features': 128, 'out_features': 10}
    },
]

# Architecture 2: One QuanvLayer, then Conv2d
test_architecture_2 = [
    {
        'layer_class': QuanvLayer,
        'params': {
            'qkernel_shape': 2,
            'embedding': default_embedding,
            'circuit': default_circuit,
            'measurement': default_measurement,
            'params': {'observable': qml.PauliZ, 'rotation': 'X', 'num_layers': 1},
            # 'out_channels' is handled automatically
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': nn.Conv2d,
        'params': {
            'out_channels': 'auto',  # Set to 'auto'
            'kernel_size': 3,
            'padding': 1
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {'layer_class': nn.Flatten, 'params': {}},
    {
        'layer_class': nn.Linear,
        'params': {'in_features': 'auto', 'out_features': 128}
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': nn.Linear,
        'params': {'in_features': 128, 'out_features': 10}
    },
]

# Architecture 3: Two QuanvLayers, then Conv2d
test_architecture_3 = [
    {
        'layer_class': QuanvLayer,
        'params': {
            'qkernel_shape': 2,
            'embedding': default_embedding,
            'circuit': default_circuit,
            'measurement': default_measurement,
            'params': {'observable': qml.PauliZ, 'rotation': 'X', 'num_layers': 1},
            # 'out_channels' is handled automatically
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': QuanvLayer,
        'params': {
            'qkernel_shape': 3,
            'embedding': default_embedding,
            'circuit': default_circuit,
            'measurement': default_measurement,
            'params': {'observable': qml.PauliZ, 'rotation': 'X', 'num_layers': 1},
            # 'out_channels' is handled automatically
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': nn.Conv2d,
        'params': {
            'out_channels': 'auto',  # Set to 'auto'
            'kernel_size': 3,
            'padding': 1
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {'layer_class': nn.Flatten, 'params': {}},
    {
        'layer_class': nn.Linear,
        'params': {'in_features': 'auto', 'out_features': 128}
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': nn.Linear,
        'params': {'in_features': 128, 'out_features': 10}
    },
]

# Architecture 4: Conv2d, then QuanvLayer
test_architecture_4 = [
    {
        'layer_class': nn.Conv2d,
        'params': {
            'out_channels': 'auto',  # Set to 'auto'
            'kernel_size': 3,
            'padding': 1
            # 'in_channels' is handled automatically by QuanvolutionalNet
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': QuanvLayer,
        'params': {
            'qkernel_shape': 2,
            'embedding': default_embedding,
            'circuit': default_circuit,
            'measurement': default_measurement,
            'params': {'observable': qml.PauliZ, 'rotation': 'X', 'num_layers': 1},
            # 'out_channels' is handled automatically
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {'layer_class': nn.Flatten, 'params': {}},
    {
        'layer_class': nn.Linear,
        'params': {'in_features': 'auto', 'out_features': 128}
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': nn.Linear,
        'params': {'in_features': 128, 'out_features': 10}
    },
]
