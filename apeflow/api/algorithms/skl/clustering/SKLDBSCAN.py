# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import numpy as np
from sklearn.cluster import DBSCAN

from apeflow.common.Constants import Constants
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract


class SKLDBSCAN(SKLAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLDBSCAN"
    ALG_TYPE = ["Clustering"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        if wrapper is not None:
            super(SKLDBSCAN, self).__init__(param_dict, wrapper, ext_data)
        else:
            super(SKLDBSCAN, self).__init__(param_dict, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(SKLDBSCAN, self)._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["eps"] = float(param_dict["eps"])
            _param_dict["min_samples"] = int(param_dict["min_samples"])
        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        eps = self.param_dict["eps"]
        min_samples = int(self.param_dict["min_samples"])

        # linear, poly, rbf, sigmoid, precomputed
        self.model = DBSCAN(
            eps=eps, min_samples=min_samples
            )

    def learn(self, dataset):
        self.model.fit(X=dataset["x"])
        self.learn_result(dataset)

    def predict(self, x):
        data_x = x
        nr_samples = data_x.shape[0]

        y_new = np.ones(shape=nr_samples, dtype=int) * -1

        for i in range(nr_samples):
            diff = self.model.components_ - data_x[i, :]  # NumPy broadcasting

            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

            shortest_dist_idx = np.argmin(dist)

            if dist[shortest_dist_idx] < self.model.eps:
                y_new[i] = self.model.labels_[self.model.core_sample_indices_[shortest_dist_idx]]

        return y_new


if __name__ == '__main__':

    __dataset = {
        "x": np.array([[-1, -1], [-2, -1], [1, 1], [2, 1],
                       [-1, -1], [-2, -1], [1, 1], [2, 1],
                       ]),
        # "y" : np.array([1, 1, 0, 0]),
    }

    __param_dict = {
        "algorithm_code": "SKLKMeans",
        "algorithm_type": "Clustering",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "1",
        "global_step": "1000",
        "model_nm": "SKLDBSCAN_2",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",

        "early_type": "0",
        "job_key": "340u0321",
        "params": {
            "eps": "0.5",
            "min_samples": "2"
        }
    }

    GSSG = SKLDBSCAN(__param_dict, None)
    GSSG._build()

    GSSG.learn(dataset=__dataset)
    GSSG.learn_result(__dataset)
    print(GSSG.predict(np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])))

    GSSG.saved_model()

    temp = SKLDBSCAN(__param_dict, None)
    temp.load_model()

    eval_data = {"x": np.array([[1., 1.]])}
    print(temp.eval(dataset=eval_data))
