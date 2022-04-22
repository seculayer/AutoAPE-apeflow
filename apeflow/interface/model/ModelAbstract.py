# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.
from typing import Union

from apeflow.common.Common import Common
from apeflow.api.algorithms.AlgorithmFactory import AlgorithmFactory
from apeflow.api.algorithms.gs.GSAlgAbstract import GSAlgAbstract
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract
from apeflow.api.algorithms.pytorch import PyTorchAlgAbstract


class ModelAbstract(object):
    def __init__(self, param_dict: dict, ext_data=None):
        self.LOGGER = Common.LOGGER.getLogger()
        self.param_dict = param_dict
        self.ext_data = ext_data
        self.model = None

    def _build(self) -> Union[GSAlgAbstract, TFKerasAlgAbstract, SKLAlgAbstract, PyTorchAlgAbstract]:
        model = AlgorithmFactory.create(param_dict=self.param_dict, ext_data=self.ext_data)
        model.load_model()

        return model

    def learn(self, dataset):
        # learning
        self.model.learn(dataset)
        self.model.saved_model()

    def eval(self, dataset):
        # learning
        result = self.model.eval(dataset)
        return result

    def predict(self, x):
        result = self.model.predict(x)
        return result
