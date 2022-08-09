# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract


class SKLExtraTrees(SKLAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLExtraTrees"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        if wrapper is not None:
            super(SKLExtraTrees, self).__init__(param_dict, wrapper, ext_data)
        else:
            super(SKLExtraTrees, self).__init__(param_dict, ext_data)

    def _build(self):
        self.model = ExtraTreesClassifier(verbose=0)

    def learn(self, dataset):
        self.model.fit(dataset["x"], self._arg_max(dataset["y"]))
        self.learn_result(dataset)


if __name__ == '__main__':
    __dataset = {
        "x": np.array([[-1., -1.], [-2., -1.], [1., 1.], [2., 1.]]),
        # "y" : np.array([1, 1, 0, 0]),
        "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0]]),
    }

    __param_dict = {
        "algorithm_code": "SKLExtraTrees",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "global_step": "1000",
        "model_nm": "SKLExtraTrees__1",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "job_key": "145123513",

        "early_type": "0",
        "params": {}
    }

    GSSG = SKLExtraTrees(__param_dict, None)
    GSSG._build()

    GSSG.learn(dataset=__dataset)

    GSSG.saved_model()

    temp = SKLExtraTrees(__param_dict, None)
    temp.load_model()

    eval_data = {"x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
                 "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0]]), }

    print(temp.eval(dataset=eval_data))

    print(temp.predict(np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])))
