# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import numpy as np
from sklearn.decomposition import PCA

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract


class SKLPCA(SKLAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLPCA"
    ALG_TYPE = ["FE"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(SKLPCA, self).__init__(param_dict, wrapper, ext_data)

    def _build(self):
        n_components = int(self.param_dict["output_units"])
        # linear, poly, rbf, sigmoid, precomputed
        self.model = PCA(
            n_components=n_components
            )

    def learn(self, dataset):
        self.model.fit(X=dataset["x"])
        self.learn_result(dataset)

    def predict(self, x):
        predict_result = self.model.transform(x)

        return predict_result


if __name__ == '__main__':

    __dataset = {
        "x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
        # "y" : np.array([1, 1, 0, 0]),
    }

    __param_dict = {
        "algorithm_code": "SKLPCA",
        "algorithm_type": "FE",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "global_step": "1000",
        "model_nm": "SKLPCA_242",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",


        "early_type": "0",
        "job_key": "310500004331",
        "params": {}
    }

    GSSG = SKLPCA(__param_dict, None)

    GSSG._build()

    GSSG.learn(dataset=__dataset)
    GSSG.learn_result(__dataset)
    print(GSSG.predict(np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])))

    GSSG.saved_model()

    temp = SKLPCA(__param_dict, None)
    temp.load_model()

    eval_data = {"x": np.array([[1., 1.]])}
    print(temp.eval(dataset=eval_data))
