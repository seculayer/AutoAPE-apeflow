# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import numpy as np
from sklearn.cluster import KMeans

from apeflow.common.Constants import Constants
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract


class SKLODBKM(SKLAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLODBKM"
    ALG_TYPE = ["OD"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(SKLODBKM, self).__init__(param_dict, wrapper, ext_data)

        input_units = self.param_dict["input_units"]
        self.r = self.get_euclidean_distance([0.] * input_units[0], [1.] * input_units[0])

    def _check_parameter(self, param_dict):
        _param_dict = super(SKLODBKM, self)._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["num_cluster"] = 1
        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        num_cluster = self.param_dict["num_cluster"]
        global_step = self.learn_params["global_step"]

        self.model = KMeans(
            n_clusters=num_cluster, verbose=0,
            random_state=0, max_iter=global_step)

    def learn(self, dataset):
        self.model.fit(X=dataset["x"])
        self.learn_result(dataset)

    def predict(self, x):
        distance = self.model.transform(x)  # return numpy array
        distance = np.squeeze(distance)
        result = np.where(distance >= self.r, 1, 0)
        return result

    @staticmethod
    def get_euclidean_distance(point, centroid):
        sum_sq = 0.0

        if len(np.shape(point)) == 1 and len(np.shape(centroid)) == 1:
            for i in range(len(point)):
                sum_sq += (point[i] - centroid[i]) ** 2

            d = sum_sq ** 0.5

        elif len(np.shape(point)) == 2 and len(np.shape(centroid)) == 2:
            d = list()
            for i in range(len(point)):
                for j in range(len(point[0])):
                    sum_sq += (point[i][j] - centroid[0][j]) ** 2

                d_row = sum_sq ** 0.5
                d.append(d_row)
        else:
            d = -100
        return d


if __name__ == '__main__':
    __dataset = {
        "x": np.array([[1, 1], [2, 2], [1, 1], [2, 1]]),
        # "y" : np.array([1, 1, 0, 0]),
    }

    __param_dict = {
        "algorithm_code": "SKLODBKM",
        "algorithm_type": "OD",
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
        "params": {
            "num_cluster": "1"
        },
        "job_key": "359209504",

        "early_type": "0"
    }

    GSSG = SKLODBKM(__param_dict, None)
    GSSG._build()

    GSSG.learn(dataset=__dataset)
    GSSG.learn_result(__dataset)
    print(GSSG.predict(np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])))

    GSSG.saved_model()

    temp = SKLODBKM(__param_dict, None)
    temp.load_model()

    eval_data = {"x": np.array([[1., 1.], [2., 2.], [100, 100]])}
    print(temp.eval(dataset=eval_data))
