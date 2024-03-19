# -*- coding: utf-8 -*-
# Author : IlJu Mun
# e-mail : ilju.mun@seculayer.com
# Powered by Seculayer © 2021 AI Service Model Team, R&D Center.

# ---- python base packages
import json
from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from pycmmn.utils.FileUtils import FileUtils
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    AveragePooling1D,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

from apeflow.api.algorithms.tf.keras.nn.KCNN import KCNN
from apeflow.interface.utils.tf.TFUtils import TFUtils
from apeflow.common.Constants import Constants


class KTMSSiameseNetworkBackborn(KCNN):
    # MODEL INFORMATION
    ALG_CODE = "KTMSBackborn"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "2.0.0"

    def __init__(self, param_dict, **kwargs):
        self.resource_model_path = f"{Constants.DIR_RESOURCES_MODEL}/tms/{self.VERSION}"
        super(KTMSSiameseNetworkBackborn, self).__init__(self.backborn_param_load(), **kwargs)
        self.siamese_net = self._build_siamese_net()

    def _check_parameter(self, param_dict):
        _param_dict = super(KTMSSiameseNetworkBackborn, self)._check_parameter(param_dict)
        return _param_dict

    def _build(self):
        super(KTMSSiameseNetworkBackborn, self)._build()
        self.backborn_load_model()

    # -- SiameseNetwork build
    def _build_siamese_net(self) -> tf.keras.Model:
        input_a = Input(shape=self.param_dict["input_units"], name="input_a")
        output_a = self.model(input_a)

        input_b = Input(shape=self.param_dict["input_units"], name="input_b")
        output_b = self.model(input_b)

        output = Lambda(TFUtils.euclidean_distance, name="distance_layer")(
            [output_a, output_b]
        )
        output = Dense(1, activation="relu")(output)

        model = Model([input_a, input_b], output)
        model.summary(print_fn=self.LOGGER.info)
        # if FileUtils.is_exist(self.sn_model_path) and not self.model_init:
        #     model.load_weights(self.sn_model_path)

        learning_rate = self.param_dict.get("learning_rate", 0.001)
        # optimizer = Adam(learning_rate)
        optimizer = Adadelta(learning_rate)
        # optimizer = RMSprop(learning_rate)
        # optimizer = "sgd"
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=optimizer)
        # model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)

        return model

    @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs = tf.keras.backend.cast(inputs, tf.float32)
        model = self.model(inputs)
        return model

    def eval_clustering(self, dataset):
        raise NotImplementedError

    def eval_we(self, dataset):
        raise NotImplementedError

    def backborn_param_load(self) -> Dict:
        # Model Parameter Load
        f = open(f"{self.resource_model_path}/info/KTMSSiameseNetworkBackborn-param.json", "r")
        param_dict = json.loads(f.read())
        f.close()
        return param_dict

    def learn(self, dataset, global_step):
        features = dataset.get("x")
        labels = dataset.get("y")
        self.siamese_net.fit(
            [features[:, 0], features[:, 1]],
            labels,
            batch_size=128,
            epochs=global_step,
            verbose=1,
            # shuffle=True,
        )

    def backborn_saved_model(self):
        model_path = f"{self.resource_model_path}/KTMSSiameseNetworkBackborn.h5"
        self.model.save(model_path)
        self.model.save_weights(
            f"{self.resource_model_path}/KTMSSiameseNetworkBackborn-weight.h5"
        )
        self.LOGGER.info("back-born_model_saved...")

    def backborn_load_model(self):
        model_path = f"{self.resource_model_path}/KTMSSiameseNetworkBackborn-weight.h5"
        if FileUtils.is_exist(model_path):
            # self.model = tf.keras.models.load_model(
            # Constants.DIR_KERAS_BACKBORN + "/KIDPSCNNBackborn-{}.h5".format(self.VERSION))
            self.model.load_weights(model_path)
            self.LOGGER.info("back-born_model_loaded...")
            self.predicts = self.model


if __name__ == '__main__':
    backborn = KTMSSiameseNetworkBackborn(param_dict={})
