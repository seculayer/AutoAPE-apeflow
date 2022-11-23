# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf

from apeflow.common.Common import Common
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract
from apeflow.interface.utils.tf.TFUtils import TFUtils


class KDNN(TFKerasAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "KDNN"
    ALG_TYPE = ["Classifier", "Regressor"]
    DATA_TYPE = ["Single"]
    VERSION = "2.0.0"

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(KDNN, self).__init__(param_dict, wrapper, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(KDNN, self)._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["hidden_units"] = list(map(int, str(param_dict["hidden_units"]).split(",")))
            # _param_dict["initial_weight"] = float(param_dict["initial_weight"])
            _param_dict["act_fn"] = str(param_dict["act_fn"])
            _param_dict["model_nm"] = str(param_dict["model_nm"])
            _param_dict["alg_sn"] = str(param_dict["alg_sn"])
            _param_dict["algorithm_type"] = str(param_dict["algorithm_type"])
            _param_dict["dropout_prob"] = float(param_dict["dropout_prob"])
            _param_dict["learning_rate"] = float(param_dict["learning_rate"])
            _param_dict["optimizer_fn"] = str(param_dict["optimizer_fn"])
        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        # KERAS GRAPH
        # Parameter Setting
        input_units = self.param_dict["input_units"]
        output_units = self.param_dict["output_units"]
        hidden_units = self.param_dict["hidden_units"]
        # initial_weight = self.param_dict["initial_weight"]
        act_fn = self.param_dict["act_fn"]
        dropout_prob = self.param_dict["dropout_prob"]
        optimizer_fn = self.param_dict["optimizer_fn"]
        learning_rate = self.param_dict["learning_rate"]

        activation = eval(Common.ACTIVATE_FN_CODE_DICT[act_fn])
        flatten_units = 1
        for ele in input_units:
            flatten_units *= ele
        units = TFUtils.get_units(flatten_units, hidden_units, output_units)

        model_nm = "{}_{}".format(self.param_dict["model_nm"], self.param_dict["alg_sn"])

        self.model = tf.keras.Sequential()

        # MAKE INPUT LAYER
        self.inputs = tf.keras.Input(shape=input_units, name=model_nm + '_X')
        self.model.add(self.inputs)

        if len(input_units) > 1:
            self.model.add(tf.keras.layers.Flatten())

        # MAKE MLP LAYERS
        TFUtils.tf_keras_mlp_block_v2(
            self.model, units, activation,
            dropout_prob=dropout_prob, name=model_nm, alg_type=self.param_dict["algorithm_type"]
        )
        self.predicts = self.model.get_layer(index=-1)

        # MAKE TRAINING METRICS
        if self.param_dict["algorithm_type"] == "Classifier":
            loss_fn_nm = 'categorical_crossentropy'
            if self.param_dict["output_units"] == 1:
                loss_fn_nm = "binary_crossentropy"
            self.model.compile(
                loss=loss_fn_nm,
                optimizer=eval(Common.OPTIMIZER_FN_CODE_DICT[optimizer_fn])(learning_rate),
                metrics=['accuracy']
            )

        elif self.param_dict["algorithm_type"] == "Regressor":
            self.model.compile(
                loss="mse",
                optimizer=eval(Common.OPTIMIZER_FN_CODE_DICT[optimizer_fn])(learning_rate),
            )

        self.model.summary(print_fn=self.LOGGER.info)


if __name__ == '__main__':
    # CLASSIFIER

    physical_devices = tf.config.list_physical_devices('GPU')
    print("physical devices: ", physical_devices)
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    __param_dict = {
        "algorithm_code": "KDNN",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "hidden_units": "5,4,3",
        "global_step": "100",
        "dropout_prob": "0.5",
        "optimizer_fn": "Adam",
        "model_nm": "KDNN-1111111111111111",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "learning_rate": "0.01",
        "initial_weight": "0.1",
        "num_layer": "5",
        "act_fn": "ReLU",

        "early_type": "0",
        "minsteps": "10",
        "early_key": "accuracy",
        "early_value": "0.98"
    }

    import numpy as np

    dataset = {
        "x": np.array([[-1., -1.], [-2., -1.], [1., 1.], [2., 1.]]),
        "y": np.array([[0.5, 0.5], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]),
    }

    GSSG = KDNN(__param_dict)
    GSSG._build()

    GSSG.learn(dataset=dataset)

    GSSG.saved_model()

    temp = KDNN(__param_dict)
    temp.load_model()

    eval_data = {"x": [[3., 2.]], "y": np.array([[1., 0.]])}
    temp.eval(eval_data)
