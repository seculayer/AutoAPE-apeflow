# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import numpy as np
from sklearn.cluster import KMeans

from apeflow.common.Constants import Constants
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract


class SKLKMeans(SKLAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLKMeans"
    ALG_TYPE = ["Clustering"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        if wrapper is not None:
            super(SKLKMeans, self).__init__(param_dict, wrapper, ext_data)
        else:
            super(SKLKMeans, self).__init__(param_dict, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(SKLKMeans, self)._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["num_cluster"] = str(param_dict["num_cluster"])
        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        num_cluster = int(self.param_dict["num_cluster"])
        global_step = int(self.learn_params["global_step"])

        # linear, poly, rbf, sigmoid, precomputed
        self.model = KMeans(
            n_clusters=num_cluster, verbose=0,
            random_state=0, max_iter=global_step)

    def learn(self, dataset):
        self.model.fit(X=dataset["x"])
        self.learn_result(dataset)


if __name__ == '__main__':
    __dataset = {
        "x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]),
        # "y" : np.array([1, 1, 0, 0]),
    }

    __param_dict = {
        "algorithm_code": "SKLKMeans",
        "algorithm_type": "Clustering",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "1",
        "global_step": "100",
        "model_nm": "SKLKMeans_224",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "job_key": "3250299999",
        "params": {
            "num_cluster": "2",
        },

        "early_type": "0"
    }

    GSSG = SKLKMeans(__param_dict, None)
    GSSG._build()

    GSSG.learn(dataset=__dataset)
    GSSG.learn_result(__dataset)
    print(GSSG.predict(np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])))

    GSSG.saved_model()

    temp = SKLKMeans(__param_dict, None)
    temp.load_model()

    eval_data = {"x": np.array([[1., 1.]])}
    print(temp.eval(dataset=eval_data))
