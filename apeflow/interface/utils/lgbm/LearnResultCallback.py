# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2022 AI Service Model Team, R&D Center.
import json
import os
from typing import List

import lightgbm

from apeflow.common.Common import Common
from apeflow.common.Constants import Constants
from pycmmn.rest.RestManager import RestManager


class LearnResultCallback(object):
    def __init__(self, **kwargs):
        self.job_key = kwargs["job_key"]
        self.global_sn = kwargs["global_sn"]
        self.data_len = kwargs["data_len"]

        self.learn_result: List = list()
        self.LOGGER = Common.LOGGER.getLogger()

    def eval_callback(self):
        def _callback(env: lightgbm.callback.CallbackEnv):
            metric = env.evaluation_result_list[0][2]
            result = {
                "global_sn": self.global_sn,
                "step": env.iteration,
                "loss": metric
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

        return _callback

    def get_learn_result(self):
        return self.learn_result
