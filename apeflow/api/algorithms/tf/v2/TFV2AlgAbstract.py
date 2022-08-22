# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import json
import os
import tensorflow as tf

from apeflow.common.Constants import Constants
from apeflow.interface.model.export.TFSavedModel import TFSavedModel
from apeflow.api.algorithms.AlgorithmAbstract import AlgorithmAbstract


class TFV2AlgAbstract(AlgorithmAbstract, tf.keras.models.Model):
    # MODEL INFORMATION (STATIC)
    ALG_CODE = "TFV2AlgAbstract"
    ALG_TYPE = []
    DATA_TYPE = []
    VERSION = "2.0.0"
    LIB_TYPE = Constants.TF
    TAG = "serve"
    OUT_MODEL_TYPE = Constants.OUT_MODEL_TF

    def __init__(self, param_dict, ext_data=None):
        tf.keras.models.Model.__init__(self)
        super(TFV2AlgAbstract, self).__init__(param_dict, ext_data)
        # AlgorithmAbstract.__init__(self, param_dict, ext_data=ext_data)

        self.num_workers = param_dict["num_workers"]

        self.input_name = "{}_{}_inputs".format(param_dict["model_nm"], param_dict["alg_sn"])
        self.output_name = "{}_{}_predicts".format(param_dict["model_nm"], param_dict["alg_sn"])

        # VARIABLES
        self.model = None
        try:
            self.task_idx = int(json.loads(os.environ["TF_CONFIG"])["task"]["index"])
        except:
            self.task_idx = 0

        # -- build model
        self._build()

    def _build(self):
        raise NotImplementedError

    def saved_model(self):
        if self.task_idx == 0:
            TFSavedModel.save(self)

    def load_model(self):
        TFSavedModel.load(self)

    def learn(self, dataset):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
