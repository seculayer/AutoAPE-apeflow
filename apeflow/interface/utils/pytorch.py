from typing import Callable, Dict, Optional, Type, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset

T = TypeVar("T", Type[nn.Module], Callable[..., torch.Tensor])

activation_map: Dict[str, Dict[str, T]] = {
    "nn": {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
    },
    "f": {
        "ReLU": F.relu,
        "Tanh": F.tanh,
        "Sigmoid": F.sigmoid,
    },
}

optimizer_mapping: Dict[str, Type[optim.Optimizer]] = {
    "Adam": optim.Adam,
    "Adadelta": optim.Adadelta,
    "rmsprop": optim.RMSprop,
}

convolution_mapping: Dict[str, Type[nn.modules.conv._ConvNd]] = {
    "Conv1D": nn.Conv1d,
    "Conv2D": nn.Conv2d,
    "Conv3D": nn.Conv3d,
}

pooling_map: Dict[str, Dict[str, T]] = {
    "nn": {
        "Max1D": nn.MaxPool1d,
        "Max2D": nn.MaxPool2d,
        "Max3D": nn.MaxPool3d,
        "Average1D": nn.AvgPool1d,
        "Average2D": nn.AvgPool2d,
        "Average3D": nn.AvgPool3d,
    },
    "f": {
        "Max1D": F.max_pool1d,
        "Max2D": F.max_pool2d,
        "Max3D": F.max_pool3d,
        "Average1D": F.avg_pool1d,
        "Average2D": F.avg_pool2d,
        "Average3D": F.avg_pool3d,
    },
}

upsampling_fn_map = {
    # TODO: PyTorch에는 Upsample이라는 class로 1~4D까지 처리한다.
    "UpSampling1D": "tf.keras.layers.UpSampling1D",
    "UpSampling2D": "tf.keras.layers.UpSampling2D",
    "UpSampling3D": "tf.keras.layers.UpSampling3D",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def createNumpyDataset(x: np.array, y: Optional[np.array] = None) -> Dataset:
    x_tensor = torch.Tensor(x).float()
    if y is not None:
        y_tensor = torch.Tensor(y).float()
        return TensorDataset(x_tensor, y_tensor)
    else:
        return TensorDataset(x_tensor)
