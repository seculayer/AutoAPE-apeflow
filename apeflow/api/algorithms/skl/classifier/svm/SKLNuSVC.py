# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import numpy as np
from sklearn.svm import NuSVC

from apeflow.common.Constants import Constants
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract


class SKLNuSVC(SKLAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLNuSVC"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(SKLNuSVC, self).__init__(param_dict, wrapper, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(SKLNuSVC, self)._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["kernel_fn"] = str(param_dict["kernel_fn"])

        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        kernel_fn = self.param_dict["kernel_fn"]

        self.model = NuSVC(kernel=kernel_fn, verbose=0, probability=True)

    def learn(self, dataset):
        # linear, poly, rbf, sigmoid, precomputed
        self.model.fit(dataset["x"], self._arg_max(dataset["y"]))
        self.learn_result(dataset)


if __name__ == '__main__':
    __dataset = {
        "x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
        # "y" : np.array([1, 1, 0, 0]),
        "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0]]),
    }

    __param_dict = {
        "algorithm_code": "SKLNuSVC",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "global_step": "1000",
        "model_nm": "SKLNuSVC__1",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",

        "early_type": "0",
        "job_key": "3155685332",
        "params": {
            "kernel_fn": "rbf",
        }
    }

    GSSG = SKLNuSVC(__param_dict, None)
    GSSG._build()

    GSSG.learn(dataset=__dataset)
    GSSG.learn_result(__dataset)
    print(GSSG.predict(np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])))

    GSSG.saved_model()

    temp = SKLNuSVC(__param_dict, None)
    temp.load_model()

    eval_data = {"x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
                 "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0]]), }

    print(GSSG.eval(dataset=eval_data))
