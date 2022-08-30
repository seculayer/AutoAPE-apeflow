# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : manki.baek@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf

from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract
from apeflow.common.Common import Common
from apeflow.interface.utils.tf.TFUtils import TFUtils
from pycmmn.exceptions.ParameterError import ParameterError


class KCNN(TFKerasAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "KCNN"
    ALG_TYPE = ["Classifier", "Regressor"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        if wrapper is not None:
            super(KCNN, self).__init__(param_dict, wrapper, ext_data)
        else:
            super(KCNN, self).__init__(param_dict, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(KCNN, self)._check_parameter(param_dict)

        # Parameter Setting
        try:
            _param_dict["hidden_units"] = list(map(int, str(param_dict["hidden_units"]).split(",")))
            _param_dict["act_fn"] = str(param_dict["act_fn"])
            _param_dict["algorithm_type"] = str(param_dict["algorithm_type"])
            _param_dict["filter_sizes"] = list(map(int, str(param_dict["filter_sizes"]).split(",")))
            _param_dict["pool_sizes"] = list(map(int, str(param_dict["pool_sizes"]).split(",")))
            _param_dict["num_filters"] = int(param_dict["num_filters"])
            _param_dict["dropout_prob"] = float(param_dict["dropout_prob"])
            _param_dict["pooling_fn"] = str(param_dict["pooling_fn"])
            _param_dict["conv_fn"] = str(param_dict["conv_fn"])
            _param_dict["optimizer_fn"] = str(param_dict["optimizer_fn"])
            _param_dict["learning_rate"] = float(param_dict["learning_rate"])
            _param_dict["initial_weight"] = float(param_dict.get("initial_weight", 0.05))
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
        filter_sizes = self.param_dict["filter_sizes"]
        pool_sizes = self.param_dict["pool_sizes"]
        num_filters = self.param_dict["num_filters"]
        dropout_prob = self.param_dict["dropout_prob"]
        pooling_fn = self.param_dict["pooling_fn"]
        conv_fn = self.param_dict["conv_fn"]
        optimizer_fn = self.param_dict["optimizer_fn"]
        learning_rate = self.param_dict["learning_rate"]
        initial_weight = self.param_dict.get("initial_weight")

        activation = eval(Common.ACTIVATE_FN_CODE_DICT[act_fn])

        # Generate to Keras Model
        self.model = tf.keras.Sequential()
        self.inputs = tf.keras.Input(shape=input_units, name="{}_{}_X".format(model_nm, alg_sn))
        self.model.add(self.inputs)

        if "1D" in conv_fn:
            # input: text (None, n_feature, 1)
            conv_stride = 1
            pooling_stride = 2
            self.model.add(
                tf.keras.layers.Reshape(
                    input_units + (1,),  # tuple operation ex) (1,2) + (3,) = (1,2,3)
                    name="{}_{}_input_reshape".format(model_nm, alg_sn)
                )
            )

        elif "2D" in conv_fn:

            # input: image (None, width, height, channel)
            conv_stride = [1, 1]
            pooling_stride = [2, 2]
            # if len(input_units) == 2:
            #     self.model.add(
            #         tf.keras.layers.Reshape(
            #             input_units + (1,),  # tuple operation ex) (1,2) + (3,) = (1,2,3)
            #             name="{}_{}_input_reshape".format(model_nm, alg_sn)
            #         )
            #     )

        else:  # 3D
            conv_stride = [1, 1, 1]
            pooling_stride = [2, 2, 2]

        for i, filter_size in enumerate(filter_sizes):
            # Convolution Layer
            conv_cls = eval(Common.CONV_FN_CODE_DICT[conv_fn])(
                kernel_size=filter_size,
                filters=num_filters,
                strides=conv_stride,
                padding="SAME",
                activation=str.lower(act_fn),
                name="{}_{}_conv_{}".format(model_nm, alg_sn, i)
            )
            self.model.add(conv_cls)

            # Pooling Layer
            pooled_cls = eval(Common.POOLING_FN_CODE_DICT[pooling_fn])(
                pool_size=pool_sizes[i],
                strides=pooling_stride,
                padding='SAME',
                name="{}_{}_pool_{}".format(model_nm, alg_sn, i))
            self.model.add(pooled_cls)

        #####################################################################################
        flatten_cls = tf.keras.layers.Flatten()
        self.model.add(flatten_cls)
        if dropout_prob != 1.0:
            self.model.add(
                tf.keras.layers.Dropout(
                    dropout_prob
                )
            )

        units = TFUtils.get_units(self.model.output_shape[1], hidden_units, output_units)

        TFUtils.tf_keras_mlp_block_v2(
            self.model, units, activation, dropout_prob=self.param_dict["dropout_prob"],
            name="{}_{}".format(model_nm, alg_sn), alg_type=self.param_dict["algorithm_type"],
            init_weight=initial_weight,
        )

        self.predicts = self.model.get_layer(index=-1)

        try:
            opt_fn = eval(Common.OPTIMIZER_FN_CODE_DICT[optimizer_fn])(learning_rate)
        except:
            opt_fn = Common.OPTIMIZER_FN_CODE_DICT[optimizer_fn]

        # MAKE TRAINING METRICS
        if self.param_dict["algorithm_type"] == "Classifier":
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=opt_fn,
                metrics=['accuracy']
            )
        elif self.param_dict["algorithm_type"] == "Regressor":
            self.model.compile(
                loss="mse",
                optimizer=opt_fn,
            )

        # if dropout_prob != 0.:
        self.model.summary(print_fn=self.LOGGER.info)


if __name__ == '__main__':
    # CLASSIFIER
    physical_devices = tf.config.list_physical_devices('GPU')
    print("physical devices: ", physical_devices)
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    __param_dict = {
        "algorithm_code": "KCNN",
        "algorithm_type": "Regressor",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "hidden_units": "64,32,4",
        "global_step": "100",
        "dropout_prob": "0.2",
        "optimizer_fn": "Adadelta",
        "model_nm": "KCNN-1111111111111112234",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "learning_rate": "0.001",

        "params": {
            "num_layer": "5",
            "act_fn": "Sigmoid",
            "filter_sizes": "2,2,2",
            "pool_sizes": "2,2,2",
            "num_filters": "64",
            "pooling_fn": "Average1D",
            "conv_fn": "Conv1D",
        },

        "early_type": "0",
        "minsteps": "10",
        "early_key": "accuracy",
        "early_value": "0.98",

        "num_workers": "1",
        "job_key": "test",
        "learning": "learn",
    }

    import numpy as np
    import os
    import json
    os.environ["TF_CONFIG"] = json.dumps({
        "task": {
            "index": 0
        }
    })

    dataset = {
        "x": np.array([[-1., -1.], [-2., -1.], [1., 1.], [2., 1.]]),
        "y": np.array([[0.5, 0.5], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]),
    }

    GSSG = KCNN(__param_dict)
    GSSG._build()

    GSSG.learn(dataset=dataset)

    eval_data = {"x": np.array([[3., 2.], [-1., -1.], [-2., -1.], [1., 1.], [2., 1.]]),
                 "y": np.array([[0., 1.], [0.5, 0.5], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])}
    GSSG.eval(eval_data)
    print(GSSG.predict(eval_data["x"]))

    GSSG.saved_model()

    temp = KCNN(__param_dict)
    temp.load_model()

    temp.eval(eval_data)
    print(temp.predict(eval_data["x"]))
