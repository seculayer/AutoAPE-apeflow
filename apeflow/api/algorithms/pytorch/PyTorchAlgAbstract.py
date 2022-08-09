# -*- coding: utf-8 -*-
# Author : Wonjoon Lee
# e-mail : wonjoon.lee@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch
from apeflow.common.Constants import Constants
from apeflow.api.algorithms.AlgorithmAbstract import AlgorithmAbstract
from apeflow.interface.model.export.PyTorchSavedModel import PyTorchSavedModel
from apeflow.interface.utils import pytorch as torch_util
from torch import nn, optim
from torch.utils.data import DataLoader


class PyTorchAlgAbstract(AlgorithmAbstract, ABC):
    ALG_CODE = "PyTorchAlgAbstract"
    ALG_TYPE = []
    DATA_TYPE = []
    VERSION = "1.8.2"  # PyTorch LTS version
    LIB_TYPE = Constants.PYTORCH
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PYTORCH

    model: nn.Module
    optimizer: optim.Optimizer
    loss_fn: nn.modules.loss._Loss

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(PyTorchAlgAbstract, self).__init__(param_dict, wrapper, ext_data)
        # VARIABLES
        self.model = None

        if wrapper is None:
            self._build()
        else:
            self.load_model()

    @abstractmethod
    def _build(self):
        raise NotImplementedError

    def learn(self, data: Dict[str, Any]):
        epochs = self.learn_params["global_step"]
        size = len(data)

        dataset = torch_util.createNumpyDataset(data["x"], data["y"])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for t in range(epochs):
            epoch_loss = torch.zeros(1).to(torch_util.device)
            for batch, (x, y) in enumerate(dataloader):
                x, y = x.to(torch_util.device), y.to(torch_util.device)
                y_pred = self.model(x)

                # Compute and print loss
                loss = self.loss_fn(y_pred, y)
                epoch_loss += loss

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch: {t}, loss: {epoch_loss.item()}")

    def predict(self, data: np.array) -> np.array:
        results: List[np.array] = []
        batch_size = self.batch_size

        dataset = torch_util.createNumpyDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        self.model.eval()
        with torch.no_grad():
            for (x,) in dataloader:
                x = x.to(torch_util.device)
                results.append(self.model(x).detach().cpu().numpy())

        return np.concatenate(results)

    def saved_model(self) -> None:
        PyTorchSavedModel.save(model=self)

    def load_model(self) -> None:
        PyTorchSavedModel.load(model=self)
