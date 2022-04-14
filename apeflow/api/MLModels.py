# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, Intelligence R&D Center.

import tensorflow as tf
from typing import List, Union

from apeflow.api.algorithms.AlgorithmFactory import AlgorithmFactory
from apeflow.common.Common import Common
from apeflow.common.Constants import Constants
from apeflow.interface.ModelInterface import ModelInterface
from apeflow.interface.utils.tf.TFUtils import TFUtils
from pycmmn.exceptions.LearningError import LearningError
from pycmmn.exceptions.EvaluationError import EvaluationError
from pycmmn.exceptions.GPUMemoryError import GPUMemoryError


class MLModels(object):
    def __init__(self, param_dict_list, cluster, task_idx, **kwargs):
        self.AI_LOGGER = Common.LOGGER.getLogger()
        self.job_key: str = kwargs["job_key"]
        self.LIB_TYPE: List[str] = self._get_lib_types(param_dict_list)
        self.ALG_CODE_LIST: List = self._get_alg_code(param_dict_list)
        self.num_workers = len(cluster["worker"])
        # INITIALIZING
        self._set_backend(task_idx)

        self.param_dict_linked_list: dict = self._make_param_dict_linked_list(param_dict_list)
        self.task_idx = task_idx
        self.AI_LOGGER.debug(self.param_dict_linked_list)
        self.ext_data: dict = dict()
        self.job_type: Union[str, None] = kwargs.get("job_type", None)

        self.AI_LOGGER.info("APEFlow v.{} MLModels initialized complete. [{}]".format(Constants.VERSION, self.job_key))

    def build(self) -> None:
        temp = self.param_dict_linked_list

        while temp is not None:
            temp["interface"]: ModelInterface = self._build(temp, self.job_type, self.ext_data)
            temp = temp["next"]

    @staticmethod
    def _build(_param_dict_linked_list, job_type, ext_data) -> ModelInterface:
        return ModelInterface(_param_dict_linked_list["method_type"],
                              _param_dict_linked_list["param_dict_list"],
                              job_type,
                              ext_data)

    # LEARNING
    def learn(self, data) -> None:
        if len(data["x"]) == 0:
            self.AI_LOGGER.error("Length of Learn Dataset is Zero!!!!!!!!!!!!")
            raise NotImplementedError
        else:
            self.AI_LOGGER.info("Learn Dataset Length is [{}]".format(len(data['x'])))

        temp = self.param_dict_linked_list
        prev_data = None
        while temp is not None:
            models: ModelInterface = temp["interface"]

            # SET DATSET
            models.set_dataset(input_data=data, prev_data=prev_data)
            try:
                # LEARNING
                models.learn()

            except tf.errors.ResourceExhaustedError as e:
                self.AI_LOGGER.error(e, exc_info=True)
                raise GPUMemoryError

            except Exception as e:
                self.AI_LOGGER.error(e, exc_info=True)
                raise LearningError

            # NEXT
            if temp["next"] is not None:
                prev_data = models.predict()[0]
            temp = temp["next"]

    # EVALUATION
    def eval(self, data) -> None:
        if len(data["x"]) == 0:
            self.AI_LOGGER.error("Length of Eval Dataset is Zero!!!!!!!!!!!!")
            raise NotImplementedError
        else:
            self.AI_LOGGER.info("Eval Dataset Length is [{}]".format(len(data['x'])))

        temp = self.param_dict_linked_list
        prev_data = None
        while temp is not None:
            models: ModelInterface = temp["interface"]
            # SET DATASET
            models.set_dataset(input_data=data, prev_data=prev_data)
            try:
                models.eval()

            except tf.errors.ResourceExhaustedError as e:
                self.AI_LOGGER.error(e, exc_info=True)
                raise GPUMemoryError

            except Exception as e:
                self.AI_LOGGER.error(e, exc_info=True)
                raise EvaluationError

            # NEXT
            if temp["next"] is not None:
                prev_data = models.predict()[0]

            temp = temp["next"]

    def predict(self, data) -> list:
        temp = self.param_dict_linked_list
        prev_data = None
        result_list = None

        while temp is not None:
            models: ModelInterface = temp["interface"]
            # SET DATASET
            models.set_dataset(input_data=data, prev_data=prev_data)
            try:
                result_list = models.predict()

            except tf.errors.ResourceExhaustedError as e:
                self.AI_LOGGER.error(e, exc_info=True)
                raise GPUMemoryError

            except Exception as e:
                self.AI_LOGGER.error(e, exc_info=True)
                raise EvaluationError

            # NEXT
            if temp["next"] is not None:
                prev_data = result_list[0]

            temp = temp["next"]

        return result_list

    @staticmethod
    def _get_lib_types(param_dict_list):
        lib_type = list()
        for param_dict in param_dict_list:
            algorithm_code = param_dict["algorithm_code"]
            _lib = AlgorithmFactory.get_lib_type(algorithm_code)
            if _lib not in lib_type:
                lib_type.append(_lib)
        return lib_type

    @staticmethod
    def _get_alg_code(param_dict_list):
        alg_code_list = list()
        for param_dict in param_dict_list:
            alg_code = param_dict["algorithm_code"]
            alg_code_list.append(alg_code)

        return alg_code_list

    def _set_backend(self, task_idx):
        # tensorflow
        tf_backend_ver = TFUtils.validate_backend_tf(self.LIB_TYPE)
        if tf_backend_ver != Constants.TF_BACKEND_NONE:
            self.AI_LOGGER.info("TensorFlow Backend Type : {}".format(tf_backend_ver))
            TFUtils.tf_backend_init(self.ALG_CODE_LIST, task_idx)

    @staticmethod
    def _default_param_dict_tree(method_type, model_type):
        return {
            "method_type": method_type,
            "model_type": model_type,
            "param_dict_list": list(),
            "next": None
        }

    def _make_param_dict_linked_list(self, param_dict_list) -> dict:
        rst_dict = None
        for param_dict in param_dict_list:
            param_dict["num_workers"] = self.num_workers
            # METHOD TYPE == BASIC
            if param_dict["method_type"] == "Basic":
                rst_dict = self._make_new_linked_list(param_dict)

            # METHOD_TYPE == ATTACH
            elif param_dict["method_type"] == "ModelAttach" or param_dict["method_type"] == "DataAttach":
                temp = rst_dict
                while temp["next"] is not None:
                    temp = temp["next"]
                temp["next"] = self._make_new_linked_list(param_dict)

            # METHOD TYPE == PARALLEL
            elif param_dict["method_type"] == "Parallel":
                temp = rst_dict
                while temp["next"] is not None:
                    temp = temp["next"]
                if temp["method_type"] != "Parallel":
                    temp["next"] = self._make_new_linked_list(param_dict)
                else:
                    self._append_info(temp, param_dict)

            else:
                raise NotImplementedError

        return rst_dict

    def _make_new_linked_list(self, param_dict: dict) -> dict:
        new_tree = self._default_param_dict_tree(param_dict["method_type"], param_dict["algorithm_type"])
        self._append_info(new_tree, param_dict)

        return new_tree

    def _append_info(self, linked_list: dict, param_dict: dict):
        if AlgorithmFactory.get_lib_type(param_dict["algorithm_code"]) in Constants.TF_BACKEND_LIST:
            param_dict["session"] = tf.compat.v1.Session(graph=tf.Graph())
        param_dict["job_key"] = self.job_key

        linked_list["param_dict_list"].append(param_dict)
