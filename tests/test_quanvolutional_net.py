# tests/test_quanvolutional_net.py

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.quanvolutional_net import QuanvolutionalNet
from src.layers.quanvolution import QuanvLayer
from src.circuits.convolution.default import default_circuit
from src.circuits.embedding.default import default_embedding
from src.circuits.measurement.default import default_measurement
import pennylane as qml  # Added import for qml

from src.grids.architecture.test import (
    test_architecture_1,
    test_architecture_2,
    test_architecture_3,
    test_architecture_4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_quanvolutional_net_default_architecture():
    """
    Test QuanvolutionalNet with the default architecture.
    """
    from src.grids.architecture.default import default_architecture
    model = QuanvolutionalNet(architecture=default_architecture, n_classes=10)
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    except Exception as e:
        pytest.fail(f"QuanvolutionalNet raised an exception with default architecture: {e}")

def test_quanvolutional_net_custom_architecture():
    """
    Test QuanvolutionalNet with a custom architecture: Conv2d -> ReLU -> QuanvLayer -> ReLU -> Flatten -> Linear -> ReLU -> Linear.
    """
    architecture = [
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
            'params': {'in_features': 'auto', 'out_features': 10}
        },
    ]
    model = QuanvolutionalNet(architecture=architecture, n_classes=10)
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    except Exception as e:
        pytest.fail(f"QuanvolutionalNet raised an exception with custom architecture: {e}")

def test_quanvolutional_net_name():
    """
    Test the __name__ method of QuanvolutionalNet.
    """
    model = QuanvolutionalNet()
    assert model.__name__() == 'QuanvolutionalNet', f"Expected __name__() to return 'QuanvolutionalNet', got {model.__name__()}"

def test_quanvolutional_net_with_custom_optimizer_and_criterion():
    """
    Test QuanvolutionalNet with a custom optimizer (SGD) and loss function (CrossEntropyLoss).
    """
    architecture = [
        {'layer_class': nn.Flatten, 'params': {}},
        {
            'layer_class': nn.Linear,
            'params': {
                'in_features': 28 * 28,
                'out_features': 10
            }
        }
    ]
    model = QuanvolutionalNet(
        architecture=architecture,
        n_classes=10,
        criterion_class=nn.CrossEntropyLoss,
        optimizer_class=optim.SGD,
        optimizer_params={'lr': 0.01, 'momentum': 0.9}
    )
    X_train = torch.randn(10, 1, 28, 28)
    y_train = torch.randint(0, 10, (10,))
    try:
        model.fit(X_train=X_train, y_train=y_train, epochs=1)
        predictions = model.predict(X_train)
        assert predictions.shape == (10,), f"Expected predictions shape (10,), got {predictions.shape}"
    except Exception as e:
        pytest.fail(f"QuanvolutionalNet raised an exception during training with custom optimizer and criterion: {e}")

def test_quanvolutional_net_complex_architecture():
    """
    Test QuanvolutionalNet with a complex architecture: QuanvLayer -> ReLU -> Conv2d -> ReLU -> QuanvLayer -> ReLU -> Flatten -> Linear -> ReLU -> Linear.
    """
    model = QuanvolutionalNet(architecture=test_architecture_1, n_classes=10)
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    except Exception as e:
        pytest.fail(f"QuanvolutionalNet raised an exception with complex architecture: {e}")

def test_quanvolutional_net_one_quanv_then_conv():
    """
    Test QuanvolutionalNet with architecture: Conv2d -> ReLU -> QuanvLayer -> ReLU -> Flatten -> Linear -> ReLU -> Linear.
    """
    model = QuanvolutionalNet(architecture=test_architecture_2, n_classes=10)
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    except Exception as e:
        pytest.fail(f"QuanvolutionalNet raised an exception with one QuanvLayer then Conv2d architecture: {e}")

def test_quanvolutional_net_two_quanv_then_conv():
    """
    Test QuanvolutionalNet with architecture: QuanvLayer -> ReLU -> QuanvLayer -> ReLU -> Conv2d -> ReLU -> Flatten -> Linear -> ReLU -> Linear.
    """
    model = QuanvolutionalNet(architecture=test_architecture_3, n_classes=10)
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    except Exception as e:
        pytest.fail(f"QuanvolutionalNet raised an exception with two QuanvLayers then Conv2d architecture: {e}")

def test_quanvolutional_net_conv_then_quanv():
    """
    Test QuanvolutionalNet with architecture: Conv2d -> ReLU -> QuanvLayer -> ReLU -> Flatten -> Linear -> ReLU -> Linear.
    """
    model = QuanvolutionalNet(architecture=test_architecture_4, n_classes=10)
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    except Exception as e:
        pytest.fail(f"QuanvolutionalNet raised an exception with Conv2d then QuanvLayer architecture: {e}")
