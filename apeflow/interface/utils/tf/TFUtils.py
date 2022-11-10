# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf
import os
from apeflow.common.Constants import Constants
from apeflow.common.Common import Common
from pycmmn.exceptions.TFBackendError import TFBackendError


class TFUtils(object):

    @staticmethod
    def disable_tfv2():
        tf.compat.v1.disable_v2_behavior()
        tf.get_logger().propagate = False

    @staticmethod
    def validate_backend_tf(lib_type_list):

        # Single GPU mode
        if Constants.GPU_SINGLE in lib_type_list:
            return Constants.GPU_SINGLE

        # 정상 상태 - tfv1
        elif Constants.TFV1 in lib_type_list and (
                Constants.KERAS not in lib_type_list and Constants.TF not in lib_type_list):
            return Constants.TF_BACKEND_V1

        # 정상 상태 - tfv2
        elif Constants.TFV1 not in lib_type_list and (
                Constants.KERAS in lib_type_list or Constants.TF in lib_type_list):
            return Constants.TF_BACKEND_V2

        # 혼합 상태 - 지원 안함
        elif Constants.TFV1 in lib_type_list and (Constants.KERAS in lib_type_list or Constants.TF in lib_type_list):
            raise TFBackendError

        # TENSOR FLOW 없음
        else:
            return Constants.TF_BACKEND_NONE

    @staticmethod
    def tf_backend_init(alg_code_list, task_idx):
        # TFFactory.set_cluster(cluster, task_idx, gpu_no)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(task_idx)

        Common.LOGGER.getLogger().info("TF_CONFIG : {}".format(os.environ["TF_CONFIG"]))

        if os.environ.get("CUDA_VISIBLE_DEVICES", None) is "-1":
            Common.LOGGER.getLogger().info("Running CPU MODE")

        else:
            physical_devices = tf.config.list_physical_devices('GPU')

            if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("NVIDIA_COM_GPU_MEM_IDX", "0")

            if len(physical_devices) != 0:
                # allow growth GPU memory
                tf.config.set_visible_devices(physical_devices[0], 'GPU')

                Common.LOGGER.getLogger().info("gpu_no : {}, task_idx : {}, \
                            physical devices: {}, NVIDIA_COM_GPU_MEM_IDX : {}".format(
                    os.environ["CUDA_VISIBLE_DEVICES"], task_idx, physical_devices,
                    os.environ.get("NVIDIA_COM_GPU_MEM_IDX", "no variable!")
                ))

                if "KDAEOD" in alg_code_list:
                    gpu_weight = 0.7
                else:
                    gpu_weight = 0.35

                # 메모리 제한
                os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
                mem_limit = int(int(os.environ.get("NVIDIA_COM_GPU_MEM_POD", 1024)) * gpu_weight)
                Common.LOGGER.getLogger().info("GPU Memory Limit Size : {}".format(mem_limit))
                tf.config.experimental.set_memory_growth(physical_devices[0], False)
                tf.config.set_logical_device_configuration(
                    physical_devices[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)])
                # tf.config.set_logical_device_configuration(
                #     physical_devices[0],
                #     [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)])
            #
            else:
                Common.LOGGER.getLogger().debug("Physical Devices(GPU) are None")

    @staticmethod
    def get_units(input_units, hidden_units, output_units):
        unit_list = list()
        unit_list.append(input_units)

        for unit in hidden_units:
            unit_list.append(unit)

        unit_list.append(output_units)
        return unit_list

    @staticmethod
    def tf_keras_mlp_block_v2(model, units, activation, dropout_prob=1.0, name="mlp", alg_type="Classifier",
                              init_weight=0.05):
        for i in range(len(units) - 2):
            layer_nm = "{}_{}".format(name, str(i + 1))

            # initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
            # initailizer rollback
            initializer = tf.keras.initializers.RandomUniform(minval=-init_weight, maxval=init_weight, seed=None)
            model.add(tf.keras.layers.Dense(
                units[i + 1], activation=activation, name=layer_nm,
                kernel_initializer=initializer
            ))
            if dropout_prob != 1.0:
                model.add(tf.keras.layers.Dropout(dropout_prob))

        final_act_fn = activation
        if alg_type == "Classifier":
            final_act_fn = tf.nn.softmax
            if units[-1] == 1:
                final_act_fn = tf.nn.sigmoid

        model.add(tf.keras.layers.Dense(units[-1], activation=final_act_fn, name=name+"_predict", ))
        return model


if __name__ == '__main__':
    TFUtils.disable_tfv2()
