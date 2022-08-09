# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import numpy as np
from sklearn.neural_network import MLPClassifier

from apeflow.common.Constants import Constants
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract


class SKLMLP(SKLAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLMLP"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        if wrapper is not None:
            super(SKLMLP, self).__init__(param_dict, wrapper, ext_data)
        else:
            super(SKLMLP, self).__init__(param_dict, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(SKLMLP, self)._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["hidden_size"] = tuple(map(int, str(param_dict["hidden_size"]).split(",")))
            _param_dict["act_fn"] = str(param_dict["act_fn"])
            if _param_dict["act_fn"] == "Sigmoid":
                _param_dict["act_fn"] = "logistic"
            else:
                _param_dict["act_fn"] = str(_param_dict["act_fn"]).lower()
            _param_dict["learning_rate"] = float(param_dict["learning_rate"])

        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        hidden_size = self.param_dict["hidden_size"]
        act_fn = self.param_dict["act_fn"]
        learning_rate = self.param_dict["learning_rate"]
        global_step = self.learn_params["global_step"]

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_size,
            activation=act_fn,
            learning_rate_init=learning_rate,
            max_iter=global_step,
            verbose=0
        )

    def learn(self, dataset):
        self.model.fit(dataset["x"], y=self._arg_max(dataset["y"]))
        self.learn_result(dataset)

    def learn_result_classifier(self, dataset):
        result_list = list()
        for idx, loss in enumerate(self.model.loss_curve_):
            results = dict()
            results["global_sn"] = self.param_dict["global_sn"]
            results["accuracy"] = self.model.score(X=dataset["x"], y=self._arg_max(dataset["y"]))
            results["loss"] = loss
            results["step"] = idx + 1

            result_list.append(results)

        return result_list


if __name__ == '__main__':

    __dataset = {
        "x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
        # "y" : np.array([1, 1, 0, 0]),
        "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0]]),
    }

    __param_dict = {
        "algorithm_code": "SKLMLP",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "global_step": "10",
        "model_nm": "SKLMLP__1",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",

        "job_key": "125135890",
        "params": {
            "learning_rate": "0.1",
            "hidden_size": "100,1,2",
            "act_fn": "ReLU",
        },

        "early_type": "0"
    }

    GSSG = SKLMLP(__param_dict, None)
    GSSG._build()

    GSSG.learn(dataset=__dataset)
    GSSG.learn_result(__dataset)
    print(GSSG.predict(np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])))

    GSSG.saved_model()

    temp = SKLMLP(__param_dict, None)
    temp.load_model()

    eval_data = {"x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
                 "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0]]), }
    print(GSSG.eval(dataset=eval_data))
