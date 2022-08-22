# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from typing import Union, List, Tuple, Callable
import numpy as np
import os
import json

from apeflow.common.Common import Common
from apeflow.common.Constants import Constants
from pycmmn.utils.StringUtil import StringUtil
from apeflow.interface.model.ModelBuilder import ModelBuilder
from apeflow.interface.model.ModelAbstract import ModelAbstract
from pycmmn.rest.RestManager import RestManager


class ModelInterface(object):
    LOGGER = Common.LOGGER.getLogger()

    def __init__(self, method_type, param_dict_list, job_type, ext_data):
        self.method_type: str = method_type
        self.ext_data: Union[dict, list, None] = ext_data
        self.job_type = job_type
        self.model_list = self._build(param_dict_list)
        self.param_dict_list = param_dict_list
        self.input_data = dict()
        self.prev_data = None

    def _build(self, param_dict_list) -> List[Tuple[ModelAbstract, bool]]:
        model_list = list()
        for param_dict in param_dict_list:
            learn_yn = StringUtil.get_boolean(param_dict["learning"])
            if not learn_yn or self.job_type != "learn":
                try:
                    param_dict["dropout_prob"] = "0"
                except:
                    pass
            model: ModelAbstract = ModelBuilder.create(param_dict, self.ext_data)
            val: Tuple = (model, learn_yn)
            model_list.append(val)
        return model_list

    def set_dataset(self, input_data, prev_data):
        for key in input_data.keys():
            try:
                self.input_data[key] = np.array(input_data[key])
            except Exception as e:
                self.LOGGER.error(e, exc_info=True)
                self.input_data[key] = input_data[key]

        if prev_data is None:
            self.prev_data = prev_data
        else:
            self.prev_data = self._make_prev_data(prev_data)

    @staticmethod
    def _make_prev_data(prev_data) -> dict:
        return {"x": np.array(prev_data)}

    def learn(self) -> None:
        for model, is_learn in self.model_list:
            if is_learn:
                model.learn(self._make_dataset())

    def eval(self) -> None:
        result_list: List[Tuple[int, Union[list, dict]]] = list()

        for idx, (model, is_learn) in enumerate(self.model_list):
            if is_learn:
                tp = (idx, model.eval(self._make_dataset()))
                result_list.append(tp)

        for idx, result in result_list:
            RestManager.update_eval_result(
                rest_url_root=Constants.REST_URL_ROOT,
                logger=self.LOGGER,
                job_key=self.param_dict_list[idx]["job_key"],
                task_idx=json.loads(os.environ["TF_CONFIG"])["task"]["index"],
                rst=result
            )

        for idx, rst in result_list:
            self.LOGGER.info("result {} : {}".format(idx, rst))

    def predict(self) -> Union[List, None]:
        result_list = list()

        for model, is_learn in self.model_list:
            if model.model.ALG_CODE == "TFGPRMV2":
                result_list.append(model.predict(self._make_dataset()))
            else:
                result_list.append(model.predict(self._make_dataset()['x']))

        return result_list
        # # self.method_type이 "Parallel"가 아닐경우, self.model_list의 길이는 1
        # if is_rst_return and self.method_type != "Parallel":
        #     if len(result_list) >= 2:
        #         self.LOGGER.error("Never Occur into this case !!! Trace why result_list is greater than 1")
        #         raise TypeError
        #     return result_list[0]
        # else:
        #     for idx, result in enumerate(result_list):
        #         RestManager.post_inference_result(
        #             job_key=self.param_dict_list[idx]["job_key"],
        #             task_idx=json.loads(os.environ["TF_CONFIG"])["task"]["index"],
        #             global_sn=self.param_dict_list[idx]["global_sn"],
        #             rst=result
        #         )

    def _make_dataset(self) -> dict:
        case: Callable = {
            "Basic": self._make_basic_dataset,
            "ModelAttach": self._make_model_attach_dataset,
            "DataAttach": self._make_data_attach_dataset,
            "Parallel": self._make_parallel_dataset
        }.get(self.method_type, None)

        if case is None:
            self.LOGGER.error("Method Type is None, Interface Not Found...")
            raise ModuleNotFoundError

        return case()

    def _make_basic_dataset(self) -> dict:
        return self.input_data

    def _make_model_attach_dataset(self) -> dict:
        res_data = dict()
        if "y" in self.input_data.keys():
            res_data["y"] = self.input_data["y"]
        res_data["x"] = self.prev_data["x"]
        return res_data

    def _make_data_attach_dataset(self) -> dict:
        res_data = dict()
        if "y" in self.input_data.keys():
            res_data["y"] = self.input_data["y"]
        res_data["x"] = np.concatenate((self.input_data["x"], self.prev_data["x"]), axis=1)
        return res_data

    def _make_parallel_dataset(self) -> dict:
        res_data = dict()
        if "y" in self.input_data.keys():
            res_data["y"] = self.input_data["y"]
        res_data["x"] = self.prev_data["x"]
        return res_data
