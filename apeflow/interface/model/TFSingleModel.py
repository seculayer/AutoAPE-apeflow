# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.
import json
import os
from typing import List

import tensorflow as tf

from apeflow.api.algorithms.AlgorithmFactory import AlgorithmFactory
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract
from apeflow.interface.model.ModelAbstract import ModelAbstract
from apeflow.interface.utils.tf.TFUtils import TFUtils


class TFSingleModel(ModelAbstract):
    def __init__(self, param_dict: dict, ext_data=None):
        ModelAbstract.__init__(self, param_dict, ext_data)
        self.model: TFKerasAlgAbstract = self._build()
        # self.Session: tf.compat.v1.Session = self.param_dict["session"]

    def _build(self) -> TFKerasAlgAbstract:
        model = AlgorithmFactory.create(param_dict=self.param_dict, ext_data=self.ext_data)
        model.load_model()
        return model

    def learn(self, dataset):
        # with self.Session.as_default():
        self.model.learn(dataset)
        self.model.saved_model()

    def eval(self, dataset):
        # with self.Session.as_default():
        result = self.model.eval(dataset)
        return result

    def predict(self, x):
        """
        :param x: if Keras Model, x is numpy list. else if Tensorflow V2 Model, x is dict
        :return: predict results
        """
        result: List = self.model.predict(x)
        return result


if __name__ == '__main__':
    from apeflow.common.Constants import Constants
    from apeflow.common.Common import Common
    Common.TF_BACKEND_VER = Constants.GPU_SINGLE
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    _param_dict = {
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
            "n_estimators": "12",
            "max_depth": "6",
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

    TFUtils.tf_backend_init([""], task_idx=0)

    _model = TFSingleModel(_param_dict)
    # model.build()

    import numpy as np
    num_samples = 1000
    input_units = int(_param_dict["input_units"][0])
    _x = np.random.random((num_samples, input_units))
    tmp = np.array([[1] for i in range(num_samples)])
    sum_x = np.sum(_x, axis=1).reshape((-1, 1))
    y = np.where(sum_x > 2.0, tmp, 0 * tmp)
    y = np.concatenate((y, 1 - y), axis=1)
    _x = tf.cast(_x, tf.float32)
    data = {"x": _x, "y": y}
    _model.learn(data)

    print(_model.predict(data.get("x")))
