# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2022 AI Service Model Team, R&D Center.
import json

import numpy as np
import os
from xgboost import XGBClassifier

from apeflow.api.algorithms.AlgorithmAbstract import AlgorithmAbstract
from apeflow.interface.model.export.TFSavedModel import TFSavedModel
from apeflow.interface.utils.xgboost.LearnResultCallback import LearnResultCallback
from apeflow.common.Constants import Constants
from pycmmn.exceptions.ParameterError import ParameterError


class XGBoost(AlgorithmAbstract):
    # MODEL INFORMATION
    ALG_CODE = "XGBoost"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    OUT_MODEL_TYPE = Constants.OUT_MODEL_XGB
    LIB_TYPE = Constants.GPU_SINGLE

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        self.model = None
        super(XGBoost, self).__init__(param_dict=param_dict, wrapper=wrapper, ext_data=ext_data)
        self.gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))

        self._build()
        if wrapper is not None:
            self.load_model()

    def _check_parameter(self, param_dict):
        _param_dict = super(XGBoost, self)._check_parameter(param_dict)
        try:
            _param_dict["learning_rate"] = float(param_dict.get("learning_rate", 0.5))
            _param_dict["n_estimators"] = int(param_dict.get("n_estimators", 1000))
            _param_dict["max_depth"] = int(param_dict.get("max_depth", 6))

        except Exception as e:
            raise ParameterError
        return _param_dict

    def _build(self):
        learning_rate = self.param_dict.get("learning_rate")
        n_estimators = self.param_dict.get("n_estimators")
        max_depth = self.param_dict.get("max_depth")
        output_units = self.param_dict.get("output_units")

        self.model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            verbosity=0,
            objective="multi:softproba",
            # objective="binary:logistic",
            num_class=output_units
        )
        if self.gpu_idx != -1:
            self.model.set_params(
                predictor="gpu_predictor",
                gpu_id=self.gpu_idx,
                tree_method="gpu_hist",
            )

    def learn(self, dataset: dict):
        global_step = self.param_dict["n_estimators"]
        global_sn = self.param_dict["global_sn"]

        result_callback = LearnResultCallback(
            global_sn=global_sn,
            job_key=self.param_dict["job_key"],
            epochs=global_step,
            task_idx=self.task_idx,
            data_len=len(dataset.get("x"))
        )

        self.model.set_params(
            eval_metric="mlogloss",
            # eval_metric="logloss",
            callbacks=[result_callback],
        )

        self.model.fit(
            dataset.get("x"), dataset.get("y").argmax(axis=1),
            eval_set=[(dataset.get("x"), dataset.get("y").argmax(axis=1))],
            verbose=False,
        )
        # print(self.model.evals_result())

    def predict_proba(self, batch_x):
        self.model.object = "multi:softproba"
        return super(XGBoost, self).predict_proba(batch_x)

    def load_model(self):
        TFSavedModel.load(self)

    def saved_model(self):
        if self.task_idx == 0:
            TFSavedModel.save(self)


if __name__ == '__main__':
    __param_dict = {
        "algorithm_code": "XGBoost",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "model_nm": "XGBC-TEST01",
        "global_sn": "0",
        "alg_sn": "0",
        "job_key": "TEST",
        "learning": "learn",

        "input_units": (2,),
        "output_units": "1",
        "params": {
            "learning_rate": "0.5",
            "n_estimators": "1000",
            "max_depth": "6",
        },
        "global_step": "1",
        "early_type": "0",
        "minsteps": "10",
        "early_key": "accuracy",
        "early_value": "0.98",

    }

    xgb = XGBoost(param_dict=__param_dict)

    _dataset = {
        "x": np.array([[-1., -1.], [-2., -1.], [1., 1.], [2., 1.]]),
        "y": np.array([[1.0], [0.0], [1.0], [0.0]]),
    }

    xgb.learn(_dataset)
    xgb.saved_model()
    xgb.load_model()
    print(xgb.predict(_dataset.get("x")))
