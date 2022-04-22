# -*- coding: utf-8 -*-
import os
from typing import Callable, Optional, TypeVar, Union

import torch
from apeflow.common.Common import Common
from apeflow.common.Constants import Constants
from apeflow.api.algorithms.AlgorithmAbstract import AlgorithmAbstract
from apeflow.interface.model.export.SavedModelAbstract import (
    SavedModelAbstract,
)

_PYTORCH_FILENAME = "model.pt"
Model = TypeVar("Model", bound=AlgorithmAbstract)

_logger = Common.LOGGER.getLogger()


class PyTorchSavedModel(SavedModelAbstract):
    @classmethod
    def save(cls, model: Model):
        super().save(model)

    @classmethod
    def _save_case_fn(cls, model: Model) -> Optional[Callable]:
        return {
            Constants.OUT_MODEL_PYTORCH: _save_pytorch_model,
            Constants.OUT_MODEL_ONNX: _save_pytorch_onnx,
        }.get(model.OUT_MODEL_TYPE, None)

    @classmethod
    def _load_case_fn(cls, model: Model) -> Optional[Callable]:
        return {
            Constants.OUT_MODEL_PYTORCH: _load_pytorch_model,
            Constants.OUT_MODEL_ONNX: None,
        }.get(model.OUT_MODEL_TYPE, None)


def _save_pytorch_model(model: Model, dir_path: Union[str, bytes, os.PathLike]):
    torch.save(model.model, os.path.join(dir_path, _PYTORCH_FILENAME))


def _load_pytorch_model(model: Model, dir_path: Union[str, bytes, os.PathLike]):
    try:
        model.model = torch.load(os.path.join(dir_path, _PYTORCH_FILENAME))
    except Exception as e:
        _logger.error(e, exc_info=True)


def _save_pytorch_onnx(model: Model, dir_path: Union[str, bytes, os.PathLike]):
    pass


def _load_pytorch_onnx(model: Model, dir_path: Union[str, bytes, os.PathLike]):
    pass
