# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, Intelligence R&D Center.

from typing import Union

from pycmmn.tools.DynamicClassLoader import DynamicClassLoader
from apeflow.api.algorithms.AlgorithmManager import AlgorithmManager
from apeflow.api.algorithms.gs.GSAlgAbstract import GSAlgAbstract
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract
from apeflow.api.algorithms.pytorch import PyTorchAlgAbstract
from apeflow.common.Common import Common


class AlgorithmFactory(object):
    ALG_PACKAGES = AlgorithmManager.get_packages()
    LOGGER = Common.LOGGER.getLogger()

    @classmethod
    def create(cls, **kwargs) -> Union[GSAlgAbstract, TFKerasAlgAbstract, SKLAlgAbstract, PyTorchAlgAbstract]:
        algorithm_code = kwargs["param_dict"]["algorithm_code"]
        class_nm = algorithm_code
        # Dynamic class loader
        return DynamicClassLoader.load_multi_packages(cls.ALG_PACKAGES, class_nm, cls.LOGGER)(**kwargs)

    @classmethod
    def get_devices(cls, algorithm_code) -> Union[str, None]:
        try:
            return DynamicClassLoader.load_multi_packages(cls.ALG_PACKAGES, algorithm_code, cls.LOGGER).DEVICE_TYPE
        except Exception as e:
            return None

    @classmethod
    def get_lib_type(cls, algorithm_code) -> Union[str, None]:
        try:
            return DynamicClassLoader.load_multi_packages(cls.ALG_PACKAGES, algorithm_code, cls.LOGGER).LIB_TYPE
        except Exception as e:
            Common.LOGGER.getLogger().error(e, exc_info=True)
            return None


if __name__ == '__main__':
    # CLASSIFIER
    param_dict = {
        "algorithm_code": "KDNN",
        # "algorithm_code": "Test",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (5,),
        "output_units": "2",
        "hidden_units": "5,4,3",
        "optimizer_fn": "Adam",
        "global_step": "1000",
        "model_nm": "20190001",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "learning_rate": "0.01",
        "initial_weight": "0.1",
        "num_layer": "5",
        "act_fn": "ReLU",
        "dropout_prob": "0.5",

        "early_type": "0",
    }
    # print(AlgorithmFactory.create(param_dict=param_dict))
    print(DynamicClassLoader.load_module("apeflow.api.algorithms.tf.keras.nn", "KCNN"))
