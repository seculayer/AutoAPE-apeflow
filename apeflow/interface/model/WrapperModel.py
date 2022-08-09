# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf
from typing import List

from apeflow.interface.model.ModelAbstract import ModelAbstract
from apeflow.api.algorithms.AlgorithmFactory import AlgorithmFactory


class WrapperModel(ModelAbstract):
    def __init__(self, param_dict: dict, ext_data=None):
        ModelAbstract.__init__(self, param_dict, ext_data)
        self.Session: tf.compat.v1.Session = self.param_dict.get("session", None)
        self.model = AlgorithmFactory.create(
            param_dict=self.param_dict, ext_data=ext_data, wrapper=True
        )

    def predict(self, x):
        """
        :param x: if Keras Model, x is numpy list. else if Tensorflow V2 Model, x is dict
        :return: predict results
        """
        if self.Session is None:
            result: List = self.model.predict(x)
        else:
            with self.Session.as_default():
                result: List = self.model.predict(x)
        return result
