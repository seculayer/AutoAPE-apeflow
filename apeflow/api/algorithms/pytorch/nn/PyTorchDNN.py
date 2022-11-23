# -*- coding: utf-8 -*-
from typing import List, Optional, Type

from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.pytorch import PyTorchAlgAbstract
from apeflow.interface.utils import pytorch as torch_util
from torch import nn, optim


class PyTorchDNN(PyTorchAlgAbstract):
    # Model information
    ALG_CODE = "PyTorchDNN"
    ALG_TYPE = ["Classifier", "Regressor"]
    DATA_TYPE = ["SINGLE"]
    VERSION = ["1.8.2"]

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(PyTorchDNN, self).__init__(param_dict, wrapper, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(PyTorchDNN, self)._check_parameter(param_dict)

        try:
            _param_dict["hidden_units"] = list(
                map(int, str(param_dict["hidden_units"]).split(","))
            )
            # _param_dict["initial_weight"] = float(param_dict["initial_weight"])
            _param_dict["act_fn"] = str(param_dict["act_fn"])
            _param_dict["model_nm"] = str(param_dict["model_nm"])
            _param_dict["alg_sn"] = str(param_dict["alg_sn"])
            _param_dict["algorithm_type"] = str(param_dict["algorithm_type"])
            _param_dict["dropout_prob"] = float(param_dict["dropout_prob"])
            _param_dict["learning_rate"] = float(param_dict["learning_rate"])
            _param_dict["optimizer_fn"] = str(param_dict["optimizer_fn"])
        except:
            raise ParameterError

        return _param_dict

    def _build(self):
        self.model: nn.Module = _DNN(
            input_unit=self.param_dict["input_units"][0],
            output_unit=self.param_dict["output_units"],
            hidden_units=self.param_dict["hidden_units"],
            activation_name=self.param_dict["act_fn"],
            dropout_prob=self.param_dict["dropout_prob"],
        ).to(torch_util.device)

        self.LOGGER.info(str(self.model))

        optimizer_fn: Optional[
            Type[optim.Optimizer]
        ] = torch_util.optimizer_mapping.get(self.param_dict.get("optimizer_fn"))
        if optimizer_fn is None:
            optimizer_fn = optim.Adam

        self.optimizer: optim.Optimizer = optimizer_fn(
            self.model.parameters(), lr=self.param_dict["learning_rate"]
        )

        self.LOGGER.info(str(self.optimizer))

        algorithm_type = self.param_dict["algorithm_type"]

        if algorithm_type == "Classifier":
            if self.param_dict["output_units"] == 1:
                self.loss_fn = nn.BCELoss()
            else:
                # self.loss_fn = nn.CrossEntropyLoss()
                self.loss_fn = nn.BCEWithLogitsLoss()
        elif algorithm_type == "Regressor":
            self.loss_fn = nn.MSELoss()


class _DNN(nn.Module):
    def __init__(
        self,
        input_unit: int,
        output_unit: int,
        hidden_units: List[int],
        activation_name: str,
        dropout_prob: float,
    ):
        super().__init__()
        activation = torch_util.activation_map["nn"].get(activation_name)()
        dropout = nn.Dropout(p=dropout_prob)

        all_module = nn.ModuleList()

        unit_list: List[int] = [input_unit, *hidden_units, output_unit]

        for index, (in_, out) in enumerate(zip(unit_list, unit_list[1:])):
            if index != 0:
                all_module.append(activation)
                if dropout_prob == 0:
                    all_module.append(dropout)
            all_module.append(nn.Linear(in_, out))

        self.model = nn.Sequential(*all_module)

        # TODO: tuning last output

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import numpy as np

    dataset = {
        "x": np.array([[-1.0, -1.0], [-2.0, -1.0], [1.0, 1.0], [2.0, 1.0]]),
        "y": np.array([[0.5, 0.5], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]),
    }

    _param_dict = {
        "params": {},
        "algorithm_code": "PyTorchDNN",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "hidden_units": "5,4,3,2",
        "global_step": "10",
        "dropout_prob": "0.5",
        "optimizer_fn": "Adam",
        "model_nm": "PyTorchDNN-1111111111111111",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "learning_rate": "0.01",
        "initial_weight": "0.1",
        "num_layer": "5",
        "act_fn": "ReLU",
        "early_type": "0",
        "minsteps": "10",
        "early_key": "accuracy",
        "early_value": "0.98",
        # require options for debug
        "job_key": "123014185",
    }

    dnn = PyTorchDNN(_param_dict)
    dnn.learn(data=dataset)

    eval_data = {"x": np.array([[3.0, 2.0]]), "y": np.array([[1.0, 0.0]])}
    dnn.eval(eval_data)

    dnn.saved_model()

    temp = PyTorchDNN(_param_dict)
    temp.load_model()

    temp.eval(eval_data)
