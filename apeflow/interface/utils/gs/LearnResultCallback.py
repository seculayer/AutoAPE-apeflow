# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import os
import json

from apeflow.common.Common import Common
from apeflow.common.Constants import Constants
from gensim.models.callbacks import CallbackAny2Vec
from pycmmn.rest.RestManager import RestManager


class LearnResultCallback(CallbackAny2Vec):
    # Callback to print loss after each epoch.
    LOGGER = Common.LOGGER.getLogger()

    def __init__(self, job_key, data_len, global_sn="0"):
        self.epoch = 1
        self.global_sn = global_sn
        self.job_key = job_key
        self.data_len = data_len
        self.learn_result = list()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()

        _result = {"global_sn": self.global_sn, "step": self.epoch, "loss": loss}
        self.learn_result.append(_result)

        if json.loads(os.environ["TF_CONFIG"])["task"]["index"] == "0":
            RestManager.update_learn_result(
                rest_url_root=Constants.REST_URL_ROOT,
                logger=self.LOGGER,
                job_key=self.job_key,
                rst=self.learn_result
            )

        self.LOGGER.info(_result)
        self.epoch += 1

    def get_learn_result(self):
        return self.learn_result
