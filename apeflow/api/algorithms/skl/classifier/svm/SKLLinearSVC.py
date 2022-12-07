# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import numpy as np
from sklearn.svm import LinearSVC

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract


class SKLLinearSVC(SKLAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLLinearSVC"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(SKLLinearSVC, self).__init__(param_dict, wrapper, ext_data)

    def _build(self):
        self.model = LinearSVC(verbose=0)

    def learn(self, dataset):
        # linear, poly, rbf, sigmoid, precomputed
        self.model.fit(dataset["x"], self._arg_max(dataset["y"]))
        self.learn_result(dataset)

    def predict_proba(self, batch_x):
        score_val = self.model.decision_function(batch_x)

        score_val = np.where(score_val < -1, -1, score_val)
        score_val = np.where(score_val > 1, 1, score_val)
        score_val = np.subtract(score_val, 1)
        score_val = np.divide(score_val, 2)
        score_val = np.abs(score_val)

        return score_val


if __name__ == '__main__':
    __dataset = {
        "x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
        # "y" : np.array([1, 1, 0, 0]),
        "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0]]),
    }

    __param_dict = {
        "algorithm_code": "SKLLinearSVC",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "global_step": "1000",
        "model_nm": "SKLLinearSVC__1",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",

        "early_type": "0",
        "params": {},
        "job_key": "467277667"
    }

    GSSG = SKLLinearSVC(__param_dict, None)
    GSSG._build()

    GSSG.learn(dataset=__dataset)
    GSSG.learn_result(__dataset)
    print(GSSG.predict(np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])))

    GSSG.saved_model()

    temp = SKLLinearSVC(__param_dict, None)
    temp.load_model()

    eval_data = {"x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
                 "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0]]), }
    print(GSSG.eval(dataset=eval_data))
