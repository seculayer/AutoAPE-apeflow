# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.AlgorithmFactory import AlgorithmFactory
from apeflow.interface.model.GSModel import GSModel
from apeflow.interface.model.PyTorchModel import PyTorchModel
from apeflow.interface.model.SKLModel import SKLModel
from apeflow.interface.model.TFModel import TFModel


class ModelBuilder(object):
    @staticmethod
    def create(param_dict, ext_data=None):
        alg_code = param_dict["algorithm_code"]
        lib_type = AlgorithmFactory.get_lib_type(alg_code)
        if lib_type in Constants.TF_BACKEND_LIST:
            return TFModel(param_dict, ext_data)
        elif lib_type == Constants.SCIKIT_LEARN:
            return SKLModel(param_dict, ext_data)
        elif lib_type == Constants.GENSIM:
            return GSModel(param_dict, ext_data)
        elif lib_type == Constants.PYTORCH:
            return PyTorchModel(param_dict, ext_data)
