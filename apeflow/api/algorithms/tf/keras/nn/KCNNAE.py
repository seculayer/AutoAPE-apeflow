# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : manki.baek@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf
import numpy as np

from apeflow.common.Common import Common
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract


class KCNNAE(TFKerasAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "KCNNAE"
    ALG_TYPE = ["FE"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        if wrapper is not None:
            super(KCNNAE, self).__init__(param_dict, wrapper, ext_data)
        else:
            super(KCNNAE, self).__init__(param_dict, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(KCNNAE, self)._check_parameter(param_dict)

        # Parameter Setting
        try:
            _param_dict["algorithm_type"] = str(param_dict["algorithm_type"])
            _param_dict["filter_sizes"] = list(map(int, str(param_dict["filter_sizes"]).split(",")))
            _param_dict["pool_sizes"] = list(map(int, str(param_dict["pool_sizes"]).split(",")))
            _param_dict["num_filters"] = int(param_dict["num_filters"])
            _param_dict["pooling_fn"] = str(param_dict["pooling_fn"])
            _param_dict["conv_fn"] = str(param_dict["conv_fn"])
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
        filter_sizes = list(self.param_dict["filter_sizes"])
        pool_sizes = list(self.param_dict["pool_sizes"])
        num_filters = self.param_dict["num_filters"]
        pooling_fn = self.param_dict["pooling_fn"]
        conv_fn = self.param_dict["conv_fn"]
        optimizer_fn = self.param_dict["optimizer_fn"]
        learning_rate = self.param_dict["learning_rate"]

        # Generate to Keras Model
        self.model = tf.keras.Sequential()
        self.inputs = tf.keras.Input(shape=input_units[0], name="{}_{}_X".format(model_nm, alg_sn))
        self.model.add(self.inputs)

        conv_stride, pooling_stride, model_reshaped = self.model_setting(conv_fn, input_units[0], model_nm, alg_sn)
        self.model.add(model_reshaped)

        for i, filter_size in enumerate(filter_sizes):
            # Convolution Layer
            conv_cls = eval(Common.CONV_FN_CODE_DICT[conv_fn])(
                kernel_size=filter_size,
                filters=num_filters,
                strides=conv_stride,
                padding="SAME",
                # activation=activation,
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
        self.model.add(tf.keras.layers.Flatten())
        act_fn = "tanh"
        initializer = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
        self.model.add(tf.keras.layers.Dense(output_units,
                                             # activation=activation,
                                             kernel_initializer=initializer,
                                             name="{}_{}_dense".format(model_nm, alg_sn),
                                             activation=act_fn
                                             ))

        self.predicts = tf.keras.Sequential(self.model.submodules)
        # sec_input_units = self.predicts.output_shape[1:]

        self.model.add(tf.keras.layers.Dense(
                self.predicts.get_layer(index=-1).input_shape[-1],
                kernel_initializer=initializer,
                activation=act_fn

        )
        )

        target_shape = self.predicts.get_layer(index=-2).input_shape
        target_shape = target_shape[1:]

        self.model.add(tf.keras.layers.Reshape(
                        target_shape=target_shape
        )
        )

        pool_sizes.reverse()
        for i, filter_size in enumerate(list(reversed(filter_sizes))):
            # Convolution Layer
            conv_cls = eval(Common.CONV_FN_CODE_DICT[conv_fn])(
                kernel_size=filter_size,
                filters=num_filters,
                strides=conv_stride,
                padding="SAME",
                # activation=activation,
                name="{}_{}_deconv_{}".format(model_nm, alg_sn, i)
            )
            self.model.add(conv_cls)
            # tf.keras.layers.

            if "1D" in pooling_fn:
                upsampling_fn = "UpSampling1D"
            elif "2D" in pooling_fn:
                upsampling_fn = "UpSampling2D"
            else:
                upsampling_fn = "UpSampling3D"

            # Pooling Layer
            pooled_cls = eval(Common.UPSAMPLING_FN_CODE_DICT[upsampling_fn])(
                size=pool_sizes[i],
                name="{}_{}_upsample_{}".format(model_nm, alg_sn, i))
            self.model.add(pooled_cls)

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(input_units[0],
                                             # activation=activation,
                                             kernel_initializer=initializer,
                                             activation=act_fn
                                             ))

        self.model.compile(
            loss='mse',
            optimizer=eval(Common.OPTIMIZER_FN_CODE_DICT[optimizer_fn])(learning_rate),
            metrics=['mae']
        )
        self.model.summary(print_fn=self.LOGGER.info)

    @staticmethod
    def model_setting(conv_fn, input_units, model_nm, alg_sn):
        if "1D" in conv_fn:
            conv_stride = 2
            pooling_stride = 2
            rst = tf.keras.layers.Reshape(
                    input_units + (1,),
                    name="{}_{}_input_reshape".format(model_nm, alg_sn)
                )

        elif "2D" in conv_fn:
            conv_stride = [2, 2]
            pooling_stride = [2, 2]
            rst = tf.keras.layers.Reshape(
                    (1, input_units, 1),
                    name="{}_{}_input_reshape".format(model_nm, alg_sn)
                )

        else:
            conv_stride = [2, 2, 2]
            pooling_stride = [2, 2, 2]
            rst = tf.keras.layers.Reshape(
                    (-1, 1, input_units, 1),
                    name="{}_{}_input_reshape".format(model_nm, alg_sn)
                )

        return conv_stride, pooling_stride, rst


if __name__ == '__main__':
    # CLASSIFIER

    physical_devices = tf.config.list_physical_devices('GPU')
    print("physical devices: ", physical_devices)
    for gpu_no in range(4):
        tf.config.experimental.set_memory_growth(physical_devices[gpu_no], True)

    __param_dict = {
      "algorithm_code": "KCNNAE",
      "algorithm_type": "FE",
      "data_type": "Single",
      "method_type": "Basic",
      "input_units": (5,),
      "output_units": "2",

      "dropout_prob": "0.3",
      "optimizer_fn": "Adam",
      "model_nm": "KCNNAE_34",
      "alg_sn": "0",
      "job_type": "learn",
      "depth": "0",
      "global_sn": "0",
      "learning_rate": "0.01",

      "filter_sizes": "2",
      "pool_sizes": "2",
      "num_filters": "3",
      "pooling_fn": "Max1D",
      "conv_fn": "Conv1D",
      "global_step": "200",

      "early_type": "0",
      "minsteps": "10",
      "early_key": "accuracy",
      "early_value": "0.98"
    }

    # model = KCNNAE(param_dict=param_dict)
    #
    # model._build()
    #
    # print("???")

    x = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20]], dtype=np.float32)

####################################################################################
####################################################################################
    # model_cls = tf.keras.Sequential()
    # model_cls.add(tf.keras.layers.Input(shape=(5,)))
    # # print(model_cls.get_layer(index=-1).output_shape)
    # model_cls.add(tf.keras.layers.Reshape(target_shape=(5,1)))
    # print(model_cls.get_layer(index=-1).output_shape)
    # model_cls.add(tf.keras.layers.Conv1D(filters=3, kernel_size=2, padding='SAME'))
    # print(model_cls.get_layer(index=-1).output_shape)
    # model_cls.add(tf.keras.layers.MaxPool1D(padding="SAME"))
    # print(model_cls.get_layer(index=-1).output_shape)
    #
    # model_cls.add(tf.keras.layers.Flatten())
    # print(model_cls.get_layer(index=-1).output_shape)
    # model_cls.add(tf.keras.layers.Dense(2))
    # print(model_cls.get_layer(index=-1).output_shape)
    #
    # pred_cls = model_cls
    # temp_seq = tf.keras.Sequential()
    # temp_seq.add(pred_cls)
    # model_cls = temp_seq
    # #####################################################################################
    #
    # model_cls.add(tf.keras.layers.Dense(pred_cls.get_layer(index=-1).input_shape[-1]))
    # # print(model_cls.get_layer(index=-1).output_shape)
    # model_cls.add(tf.keras.layers.Reshape(target_shape=pred_cls.get_layer(index=3).input_shape[1:]))
    # # print(model_cls.get_layer(index=-1).output_shape)
    #
    # model_cls.add(tf.keras.layers.Conv1D(filters=3, kernel_size=2, padding="SAME"))
    # # print(model_cls.get_layer(index=-1).output_shape)
    # model_cls.add(tf.keras.layers.UpSampling1D(size=2))
    # # print(model_cls.get_layer(index=-1).output_shape)
    # model_cls.add(tf.keras.layers.Flatten())
    # # print(model_cls.get_layer(index=-1).output_shape)
    # model_cls.add(tf.keras.layers.Dense(5))
    # # print(model_cls.get_layer(index=-1).output_shape)
    # # model_cls.add(tf.keras.layers.Reshape(target_shape=(5,)))
    # # print(model_cls.get_layer(index=-1).output_shape)
    #
    #
    # model_cls.compile(loss='mse', optimizer='adam')
    #
    # model_cls.fit(x=x, y=x, epochs=10000)
    #
    # pred_x = np.array([[16,17,18,19,20],
    #                    [11, 12, 13, 14, 15],
    #                    [6, 7, 8, 9, 10],
    #                    [1, 2, 3, 4, 5]
    #         ], dtype=np.float32)
    # print(model_cls.predict(pred_x))
    # print(pred_cls.predict(pred_x))

    model_ = KCNNAE(param_dict=__param_dict)
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

    temp = KCNNAE(param_dict=__param_dict)
    temp.load_model()

    temp.eval({"x": pred_x})
    print(temp.model(pred_x))
    print(temp.predicts(pred_x))

    print(temp.predict(pred_x))
