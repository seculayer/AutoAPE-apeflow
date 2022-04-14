# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import os

from pycmmn.Singleton import Singleton
from pycmmn.utils.FileUtils import FileUtils
from pycmmn.tools.DynamicClassLoader import DynamicClassLoader
from apeflow.common.Common import Common


class AlgorithmManager(metaclass=Singleton):
    INFO = dict()
    LOGGER = Common.LOGGER.getLogger()

    @staticmethod
    def get_lib_path():
        return os.path.realpath(FileUtils.get_realpath(file=__file__))

    @classmethod
    def get_packages(cls):
        return DynamicClassLoader.get_packages(
            target_dir=cls.get_lib_path(),
            exclude_files=[
                "__init__.py",
                "AlgorithmFactory.py",
                "AlgorithmManager.py"
            ],
            lib_path=os.getcwd()+"/"
        )

    @classmethod
    def load_info(cls, packages, class_nm):
        algorithm_info = dict()
        algorithm_class = DynamicClassLoader.load_multi_packages(packages, class_nm, cls.LOGGER)

        # MAKE INFORMATION
        algorithm_info["algorithm_code"] = algorithm_class.ALG_CODE
        algorithm_info["algorithm_type"] = algorithm_class.ALG_TYPE
        algorithm_info["data_type"] = algorithm_class.DATA_TYPE
        algorithm_info["lib_type"] = algorithm_class.LIB_TYPE
        algorithm_info["version"] = algorithm_class.VERSION

        try:
            algorithm_info["device_type"] = algorithm_class.DEVICE_TYPE
        except:
            algorithm_info["device_type"] = None

        return algorithm_info


if __name__ == '__main__':
    pass
