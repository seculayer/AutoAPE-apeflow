# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : manki.baek@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf

from apeflow.common.Common import Common
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract
from apeflow.interface.utils.tf.TFUtils import TFUtils


class KRNN(TFKerasAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "KRNN"
    ALG_TYPE = ["Classifier", "Regressor"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(KRNN, self).__init__(param_dict, wrapper, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(KRNN, self)._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["hidden_units"] = list(map(int, str(param_dict["hidden_units"]).split(",")))
            _param_dict["cell_units"] = int(param_dict["cell_units"])
            _param_dict["act_fn"] = str(param_dict["act_fn"])
            _param_dict["model_nm"] = str(param_dict["model_nm"])
            _param_dict["alg_sn"] = str(param_dict["alg_sn"])
            _param_dict["algorithm_type"] = str(param_dict["algorithm_type"])
            _param_dict["rnn_cell"] = str(param_dict["rnn_cell"])
            _param_dict["dropout_prob"] = float(param_dict["dropout_prob"])
            _param_dict["optimizer_fn"] = str(param_dict["optimizer_fn"])
            _param_dict["learning_rate"] = float(param_dict["learning_rate"])
            _param_dict["seq_length"] = int(param_dict["seq_length"])
        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        # Parameter Setting
        input_units = self.param_dict["input_units"]
        output_units = self.param_dict["output_units"]
        hidden_units = self.param_dict["hidden_units"]
        act_fn = self.param_dict["act_fn"]
        model_nm = self.param_dict["model_nm"]
        alg_sn = self.param_dict["alg_sn"]
        cell_units = self.param_dict["cell_units"]
        rnn_cell = self.param_dict["rnn_cell"]
        dropout_prob = self.param_dict["dropout_prob"]

        optimizer_fn = self.param_dict["optimizer_fn"]
        learning_rate = self.param_dict["learning_rate"]

        activation = eval(Common.ACTIVATE_FN_CODE_DICT[act_fn])

        # Generate to Keras Model
        self.model = tf.keras.Sequential()
        self.inputs = tf.keras.Input(shape=input_units, name="{}_{}_X".format(model_nm, alg_sn))
        self.model.add(self.inputs)

        cell = None
        if rnn_cell == "RNN":
            cell = tf.keras.layers.SimpleRNN(
                units=cell_units,
                activation=activation,
                dropout=dropout_prob,
                # input_shape=(seq_length, input_units),
                name="{}_{}_cell".format(model_nm, alg_sn),
            )
        elif rnn_cell == "GRU":
            cell = tf.keras.layers.GRU(
                units=cell_units,
                activation=activation,
                dropout=dropout_prob,
                # input_shape=(seq_length, input_units),
                name="{}_{}_cell".format(model_nm, alg_sn),
            )
        elif rnn_cell == "LSTM":
            cell = tf.keras.layers.LSTM(
                units=cell_units,
                activation=activation,
                dropout=dropout_prob,
                # input_shape=(seq_length, input_units),
                name="{}_{}_cell".format(model_nm, alg_sn),
            )
        #
        # layer = tf.keras.layers.RNN(
        #     cell=cell,
        #     dtype=tf.float32,
        #     return_state=True
        # )

        self.model.add(cell)

        units = TFUtils.get_units(cell_units, hidden_units, output_units)
        # TFNNFactory.feedforward_network_keras(self.model, units, activation, self.param_dict, input_layer=False)

        model_nm = "{}_{}".format(self.param_dict["model_nm"], self.param_dict["alg_sn"])

        TFUtils.tf_keras_mlp_block_v2(
            self.model, units, activation,
            dropout_prob=self.param_dict["dropout_prob"], name=model_nm, alg_type=self.param_dict["algorithm_type"]
        )

        self.predicts = self.model.get_layer(index=-1)

        if self.param_dict["algorithm_type"] == "Classifier":
            self.model.compile(
                loss='categorical_crossentropy',
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

    physical_devices = tf.config.list_physical_devices('GPU')
    print("physical devices: ", physical_devices)

    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    __param_dict = {
        "algorithm_code": "KRNN",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "hidden_units": "64,32,4",
        "global_step": "100",
        "dropout_prob": "0.2",
        "optimizer_fn": "Adadelta",
        "model_nm": "KRNN-0",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "learning_rate": "0.1",

        "rnn_cell": "GRU",
        "cell_units": "4",
        "seq_length": "1",

        "act_fn": "Tanh",
        "early_type": "0",
        "minsteps": "100",
        "early_key": "loss",
        "early_value": "0.0002",

        "num_workers": "1"
    }

    # classifier
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # rnn = KRNN(param_dict=param_dict)
    #
    # x = [
    #     [
    #         [1,2,3,4,5],
    #         [1,2,3,4,5],
    #         [1,2,3,4,5],
    #         [1,2,3,4,5]
    #     ],
    #     [
    #         [1, 2, 3, 4, 5],
    #         [1, 2, 3, 4, 5],
    #         [1, 2, 3, 4, 5],
    #         [1, 2, 3, 4, 5]
    #     ],
    # ]
    # import numpy as np
    #
    # x = np.array(x)
    # y = np.array([[1, 0], [1, 0]])
    # rnn.model.fit(x, y=y, epochs=100)

    # regressor
    import numpy as np

    dataset = {
        "x": np.array([[[-1., -1.]], [[-2., -1.]], [[1., 1.]], [[2., 1.]]]),
        # "y": np.array([[0.5, 0.5], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]),
        "y": np.array([[1, 0], [1, 0], [1, 0], [1, 0]]),
    }

    GSSG = KRNN(__param_dict)
    GSSG._build()

    GSSG.learn(dataset=dataset)

    GSSG.saved_model()

    temp = KRNN(__param_dict)
    temp.load_model()

    eval_data = {"x": [[[3., 2.]]], "y": np.array([[1, 0]])}
    temp.eval(eval_data)
