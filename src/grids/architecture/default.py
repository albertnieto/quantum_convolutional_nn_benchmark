# src/grids/architecture/default.py

from src.layers.quanvolution import QuanvLayer
import torch.nn as nn
from src.circuits.convolution.default import default_circuit
from src.circuits.embedding.default import default_embedding
from src.circuits.measurement.default import default_measurement

default_architecture = [
    {
        'layer_class': QuanvLayer,
        'params': {
            'qkernel_shape': 2,
            'embedding': default_embedding,
            'circuit': default_circuit,
            'measurement': default_measurement,
            'params': {'observable': qml.PauliZ, 'rotation': 'X', 'num_layers': 1},
            # 'out_channels' is handled automatically in QuanvLayer
        }
    },
    {'layer_class': nn.ReLU, 'params': {}},
    {
        'layer_class': nn.Conv2d,
        'params': {
            'out_channels': 'auto',  # Set to 'auto' for automatic handling
            'kernel_size': 3,
            'padding': 1
            # 'in_channels' is handled automatically by QuanvolutionalNet
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
            # 'out_channels' is handled automatically in QuanvLayer
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
