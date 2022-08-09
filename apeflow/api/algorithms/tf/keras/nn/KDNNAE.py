# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : manki.baek@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf
import numpy as np

from apeflow.common.Common import Common
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract


class KDNNAE(TFKerasAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "KDNNAE"
    ALG_TYPE = ["FE"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        if wrapper is not None:
            super(KDNNAE, self).__init__(param_dict, wrapper, ext_data)
        else:
            super(KDNNAE, self).__init__(param_dict, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(KDNNAE, self)._check_parameter(param_dict)

        # Parameter Setting
        try:
            _param_dict["algorithm_type"] = str(param_dict["algorithm_type"])
            _param_dict["hidden_units"] = list(map(int, str(param_dict["hidden_units"]).split(",")))
            _param_dict["optimizer_fn"] = str(param_dict["optimizer_fn"])
            _param_dict["learning_rate"] = float(param_dict["learning_rate"])
        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        # Parameter Setting
        input_units = self.param_dict["input_units"]
        output_units = self.param_dict["output_units"]

        # act_fn = self.param_dict["act_fn"]
        model_nm = self.param_dict["model_nm"]
        alg_sn = self.param_dict["alg_sn"]
        hidden_units = self.param_dict["hidden_units"]
        optimizer_fn = self.param_dict["optimizer_fn"]
        learning_rate = self.param_dict["learning_rate"]

        # Generate to Keras Model
        self.model = tf.keras.Sequential()
        self.inputs = tf.keras.Input(shape=input_units[0], name="{}_{}_X".format(model_nm, alg_sn))
        self.model.add(self.inputs)

        #####################################################################################
        for idx, hidden_unit in enumerate(hidden_units):
            self.model.add(tf.keras.layers.Dense(hidden_unit,
                                                 name="{}_{}_encoder_dense_{}".format(model_nm, alg_sn, idx)
                                                 ))
            # self.model.add(tf.keras.layers.Dropout(dropout_prob))

        self.model.add(tf.keras.layers.Dense(output_units,
                                             # activation=activation,
                                             name="{}_{}_predicts".format(model_nm, alg_sn)))
        # self.model.add(tf.keras.layers.Dropout(dropout_prob))

        self.predicts = tf.keras.Sequential(self.model.submodules)

        hidden_units.reverse()
        for idx, hidden_unit in enumerate(hidden_units):
            self.model.add(tf.keras.layers.Dense(
                hidden_unit,
                name="{}_{}_decoder_dense_{}".format(model_nm, alg_sn, idx)
            )
            )
            # self.model.add(tf.keras.layers.Dropout(dropout_prob))

        self.model.add(tf.keras.layers.Dense(input_units[0],
                                             name="{}_{}_decoded_x".format(model_nm, alg_sn)
                                             ))
        self.model.compile(
            loss='mse',
            optimizer=eval(Common.OPTIMIZER_FN_CODE_DICT[optimizer_fn])(learning_rate),
            metrics=['mae']
        )

        self.model.summary(print_fn=self.LOGGER.info)


if __name__ == '__main__':
    # CLASSIFIER
    __param_dict = {
      "algorithm_code": "KDNNAE",
      "algorithm_type": "FE",
      "data_type": "Single",
      "method_type": "Basic",
      "input_units": (5,),
      "output_units": "2",

      "dropout_prob": "0.3",
      "optimizer_fn": "Adam",
      "model_nm": "KDNNAE_5",
      "alg_sn": "0",
      "job_type": "learn",
      "depth": "0",
      "global_sn": "0",
      "learning_rate": "0.01",

      "hidden_units": "4,3,2",
      "global_step": "100",

      "early_type": "0",
      "minsteps": "10",
      "early_key": "accuracy",
      "early_value": "0.98"
    }

    x = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20]], dtype=np.float32)

    model_ = KDNNAE(param_dict=__param_dict)
    # model_._build()
    model_.learn({"x": x})

    pred_x = np.array([[16, 17, 18, 19, 20],
                       [11, 12, 13, 14, 15],
                       [6, 7, 8, 9, 10],
                       [1, 2, 3, 4, 5]
                       ], dtype=np.float32)

    print(model_.model(pred_x))
    print(model_.predicts(pred_x))
    model_.saved_model()

    temp = KDNNAE(param_dict=__param_dict)
    temp.load_model()

    temp.eval({"x": pred_x})
    print(temp.model(pred_x))
    print(temp.predicts(pred_x))

    print(temp.predict(pred_x))
