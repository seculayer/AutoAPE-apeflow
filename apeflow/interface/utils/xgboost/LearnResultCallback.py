# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2022 AI Service Model Team, R&D Center.
import json
import os
from typing import List

import xgboost

from apeflow.common.Common import Common
from apeflow.common.Constants import Constants
from pycmmn.rest.RestManager import RestManager


class LearnResultCallback(xgboost.callback.TrainingCallback):
    def __init__(self, **kwargs):
        self.job_key = kwargs["job_key"]
        self.global_sn = kwargs["global_sn"]
        self.data_len = kwargs["data_len"]

        self.learn_result: List = list()
        self.LOGGER = Common.LOGGER.getLogger()

    def after_iteration(self, model, epoch, evals_log):
        loss = evals_log.get("validation_0").get("logloss")

        result = {
            "global_sn": self.global_sn,
            "step": epoch + 1,
            "loss": loss[-1]
        }

        self.LOGGER.info(result)

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
