# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf
import os
import json

from apeflow.common.Common import Common
from apeflow.common.Constants import Constants
from pycmmn.rest.RestManager import RestManager


class LearnResultCallback(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        tf.keras.callbacks.Callback.__init__(self)
        self.job_key = kwargs["job_key"]
        self.global_sn = kwargs["global_sn"]
        self.data_len = kwargs["data_len"]
        self.LOGGER = Common.LOGGER.getLogger()
        self.learn_result = list()

    def on_epoch_end(self, epoch, logs=None):
        result = logs
        result["step"] = epoch + 1
        self.LOGGER.info(result)
        result = {k: float(v) for k, v in result.items()}
        self.learn_result.append(result)

        if json.loads(os.environ["TF_CONFIG"])["task"]["index"] == "0":
            RestManager.update_learn_result(
                rest_url_root=Constants.REST_URL_ROOT,
                logger=self.LOGGER,
                job_key=self.job_key,
                rst=self.learn_result
            )

    def get_learn_result(self):
        return self.learn_result
