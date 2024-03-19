# -*- coding: utf-8 -*-
# Author : IlJu Mun
# e-mail : ilju.mun@seculayer.com
# Powered by Seculayer © 2021 AI Service Model Team, R&D Center.

# ---- python base packages
import json
from typing import Dict

import numpy as np
import tensorflow as tf

from apeflow.api.algorithms.tf.keras.nn.KCNN import KCNN
from apeflow.common.Constants import Constants
from pycmmn.utils.FileUtils import FileUtils


class KIDPSCNNBackborn(KCNN):
    VERSION = "1.0.0"

    def __init__(self, param_dict, **kwargs):
        self.bb_learn = kwargs.get("bb_learning", False)
        self.predicts = None
        super(KIDPSCNNBackborn, self).__init__(self.backborn_param_load(), **kwargs)

    def _build(self):
        super(KIDPSCNNBackborn, self)._build()

        self.backborn_load_model()
        self.predicts = self._make_feature_model()
        self.predicts.summary(print_fn=self.LOGGER.info)

    def predict(self, x):
        # data shard option 해제
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        batch_size = self.batch_size
        try:
            self.param_dict["dropout_prob"] = 0.0
        except:
            pass
        start = 0
        results = None
        len_x = len(x)

        while start < len_x:
            end = start + batch_size
            if start == 0 and batch_size < len_x:
                batch_x = tf.keras.backend.cast(x[start: end], tf.float32)
                # results = self.predicts.predict(batch_x)
                results = self.predicts(batch_x).numpy()

            elif start == 0 and batch_size >= len_x:
                batch_x = tf.keras.backend.cast(x, tf.float32)
                # results = self.predicts.predict(batch_x)
                results = self.predicts(batch_x).numpy()

            elif end >= len_x:
                batch_x = tf.keras.backend.cast(x[start:], tf.float32)
                results = np.concatenate((results, self.predicts(batch_x).numpy()), axis=0)

            else:
                batch_x = tf.keras.backend.cast(x[start:end], tf.float32)
                results = np.concatenate((results, self.predicts(batch_x).numpy()), axis=0)
            start += batch_size

            if start % (batch_size * 30) == 0:
                self.LOGGER.info("backborn batch... current : {} / end : {}".format(start, len_x))

        return results

    def eval_clustering(self, dataset):
        raise NotImplementedError

    def eval_we(self, dataset):
        raise NotImplementedError

    def backborn_saved_model(self):
        model_path = Constants.DIR_KERAS_BACKBORN + "/KIDPSCNNBackborn-{}.h5".format(self.VERSION)
        self.model.save(model_path)
        self.model.save_weights(
            Constants.DIR_KERAS_BACKBORN + "/KIDPSCNNBackborn-weight-{}.h5".format(self.VERSION)
        )
        self.LOGGER.info("back-born_model_saved...")

    def backborn_load_model(self):
        model_path = Constants.DIR_KERAS_BACKBORN + "/KIDPSCNNBackborn-weight-{}.h5".format(self.VERSION)
        if FileUtils.is_exist(model_path):
            # self.model = tf.keras.models.load_model(
            # Constants.DIR_KERAS_BACKBORN + "/KIDPSCNNBackborn-{}.h5".format(self.VERSION))
            self.model.load_weights(model_path)
            self.LOGGER.info("back-born_model_loaded...")
            self.predicts = self._make_feature_model()

    def _make_feature_model(self):
        return tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(index=-2).output)

    def backborn_param_load(self) -> Dict:
        # Model Parameter Load
        f = open("{}/KIDPSCNNBackbornParam-{}.json".format(Constants.DIR_KERAS_BACKBORN, self.VERSION), "r")
        param_dict = json.loads(f.read())
        f.close()
        return param_dict
