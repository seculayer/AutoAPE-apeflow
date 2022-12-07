# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2022 AI Service Model Team, R&D Center.
import json
import os

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
from typing import Union

from apeflow.api.algorithms.AlgorithmAbstract import AlgorithmAbstract
from apeflow.common.Constants import Constants
from apeflow.interface.model.export.TFSavedModel import TFSavedModel
from apeflow.interface.utils.lgbm.LearnResultCallback import LearnResultCallback
from pycmmn.exceptions.ParameterError import ParameterError


class LightGBM(AlgorithmAbstract):
    # MODEL INFORMATION
    ALG_CODE = "LightGBM"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    OUT_MODEL_TYPE = Constants.OUT_MODEL_LGBM
    LIB_TYPE = Constants.GPU_SINGLE

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        self.model: Union[LGBMClassifier, None] = None
        super(LightGBM, self).__init__(param_dict, wrapper=wrapper, ext_data=ext_data)

        if wrapper is None:
            self._build()
        else:
            self.load_model()

    def _check_parameter(self, param_dict):
        _param_dict = super(LightGBM, self)._check_parameter(param_dict)
        try:
            _param_dict["num_leaves"] = int(param_dict.get("num_leaves", 41))
            _param_dict["max_depth"] = int(param_dict.get("max_depth", 21))
        except Exception as e:
            raise ParameterError
        return _param_dict

    def _build(self):
        self.model = LGBMClassifier(
            num_leaves=self.param_dict["num_leaves"],
            max_depth=self.param_dict["max_depth"],
            objective="binary"
        )

    def learn(self, dataset: dict):
        global_step = self.learn_params["global_step"]
        global_sn = self.param_dict["global_sn"]

        result_callback = LearnResultCallback(
            global_sn=global_sn,
            job_key=self.param_dict["job_key"],
            epochs=global_step,
            task_idx=self.task_idx,
            data_len=len(dataset.get("x"))
        )

        self.model.fit(
            dataset.get("x"), self._arg_max(dataset.get("y")),
            eval_set=(dataset.get("x"), self._arg_max(dataset.get("y"))),
            callbacks=[result_callback.eval_callback()]
        )

    def load_model(self):
        TFSavedModel.load(self)

    def saved_model(self):
        TFSavedModel.save(self)


def eval_callback():
    def _callback(env: lightgbm.callback.CallbackEnv):
        print(env.iteration, env.evaluation_result_list)

    return _callback


if __name__ == '__main__':
    __param_dict = {
        "algorithm_code": "LightGBM",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "model_nm": "LGBM-TEST01",
        "global_sn": "0",
        "alg_sn": "0",
        "job_key": "TEST",
        "learning": "learn",

        "input_units": (2,),
        "output_units": "1",
        "params": {
            "num_leaves": "41",
            "max_depth": "21",
        },
        "global_step": "1",
        "early_type": "0",
        "minsteps": "10",
        "early_key": "accuracy",
        "early_value": "0.98",
    }
    cluster = {
        "worker": ["10.1.35.118:9305"],
        # "worker": ["192.168.2.235:9305", "192.168.2.235:9306"],
    }
    task_idx = 0
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster,
        "task": {"type": "worker", "index": int(task_idx)}
    })
    os.environ["CUDA_VISIBLE_DEVICES"] = str(task_idx)

    lgbm = LightGBM(param_dict=__param_dict)
    _dataset = {
        "x": np.array([[-1., -1.], [-2., -1.], [1., 1.], [2., 1.]]),
        "y": np.array([1.0, 0.0, 0.0, 1.0]),
    }

    lgbm.learn(_dataset)
    lgbm.saved_model()
    lgbm.load_model()
    print(lgbm.predict(_dataset.get("x")))
