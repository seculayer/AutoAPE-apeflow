# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.
import json
import os
import numpy as np
from typing import Callable, Dict, Union

from pycmmn.rest.RestManager import RestManager
from apeflow.common.Constants import Constants
from apeflow.common.Common import Common
from pycmmn.exceptions.ParameterError import ParameterError


class AlgorithmAbstract(object):
    ALG_CODE = "AlgorithmAbstract"
    ALG_TYPE = []
    DATA_TYPE = []
    VERSION = "1.0.0"
    LIB_TYPE = "None"

    def __init__(self, param_dict: dict, wrapper=None, ext_data=None):
        # 파라미터 체크
        self.LOGGER = Common.LOGGER.getLogger()
        _param_dict = dict(param_dict, **param_dict.get("params"))
        self.ext_data = dict() if ext_data is None else ext_data
        if wrapper is None:
            self.param_dict = self._check_parameter(_param_dict)
            self.learn_params = self._check_learning_parameter(_param_dict)
        else:
            self.param_dict = _param_dict
            self.learn_params = _param_dict
        self.early_steps = 0
        self.batch_size = Constants.BATCH_SIZE

        self.model = None
        try:
            self.task_idx = int(json.loads(os.environ["TF_CONFIG"])["task"]["index"])
        except Exception as e:
            self.task_idx = 0

    @staticmethod
    def _check_parameter(param_dict):
        _param_dict = dict()
        # KTMSSiameseNetworkBackborn-param.json의 input_units value str to tuple
        if isinstance(param_dict["input_units"], str):
            _param_dict["input_units"] = (int(param_dict["input_units"]), )
        else:
            _param_dict["input_units"] = tuple(map(int, param_dict["input_units"]))

        _param_dict["output_units"] = int(param_dict["output_units"])
        _param_dict["model_nm"] = str(param_dict["model_nm"])
        _param_dict["alg_sn"] = str(param_dict["alg_sn"])
        _param_dict["global_sn"] = str(param_dict["global_sn"])
        _param_dict["algorithm_type"] = str(param_dict["algorithm_type"])
        _param_dict["job_key"] = str(param_dict["job_key"])
        _param_dict["learning"] = str(param_dict["learning"])

        return _param_dict

    def _check_learning_parameter(self, param_dict):
        _param_dict = dict()
        # Parameter Setting
        try:
            _param_dict["global_step"] = int(param_dict["global_step"])
            if _param_dict["global_step"] < 1:
                _param_dict["global_step"] = 1
            _param_dict["early_type"] = param_dict["early_type"]
            if _param_dict["early_type"] != Constants.EARLY_TYPE_NONE:
                _param_dict["minsteps"] = int(param_dict["minsteps"])
                _param_dict["early_key"] = param_dict["early_key"]
                _param_dict["early_value"] = float(param_dict["early_value"])
        except Exception as e:
            self.LOGGER.error(e, exc_info=True)
            raise ParameterError
        return _param_dict

    def early_stop(self, **kwargs):
        if self.learn_params["early_type"] == Constants.EARLY_TYPE_NONE:
            return False

        results = kwargs["results"]
        key = self.learn_params["early_key"]
        if self.learn_params["early_type"] == Constants.EARLY_TYPE_MIN:
            if self.learn_params["minsteps"] < results[-1]["step"]:
                if results[-1][key] < self.learn_params["early_value"]:
                    return True

        elif self.learn_params["early_type"] == Constants.EARLY_TYPE_MAX:
            if self.learn_params["minsteps"] < results[-1]["step"]:
                if results[-1][key] > self.learn_params["early_value"]:
                    return True

        elif self.learn_params["early_type"] == Constants.EARLY_TYPE_VAR:
            try:
                if abs(results[-1][key] - results[-2][key]) < self.learn_params["early_value"]:
                    self.early_steps += 1
                else:
                    self.early_steps = 0
            except Exception as e:
                pass

            if self.early_steps >= self.learn_params["minsteps"]:
                self.LOGGER.info("------ EARLY STOP !!!!! -----")
                return True
        return False

    def learn(self, dataset: dict):
        raise NotImplementedError

    def predict(self, x) -> Dict:
        batch_size = self.batch_size
        start = 0
        results_pred: Union[np.ndarray, None] = None
        results_proba: Union[np.ndarray, None] = None
        len_x = len(x)
        is_classifier: bool = True if self.param_dict["algorithm_type"] == "Classifier" else False
        is_outlier_detection: bool = True if self.param_dict["algorithm_type"] == "OD" else False

        while start < len_x:
            end = start + batch_size
            batch_x = x[start: end]
            if start == 0:
                results_pred = self.predict_decision(batch_x)
                if is_classifier or is_outlier_detection:
                    results_proba = self.predict_proba(batch_x)
            else:
                results_pred = np.append(results_pred, self.predict_decision(batch_x), axis=0)
                if is_classifier or is_outlier_detection:
                    results_proba = np.append(results_proba, self.predict_proba(batch_x), axis=0)
            start += batch_size

            if self.param_dict["learning"] == "N" and (is_classifier or is_outlier_detection):
                progress_rate = start / len_x * 100
                RestManager.send_inference_progress(
                    rest_url_root=Constants.REST_URL_ROOT,
                    logger=self.LOGGER,
                    prograss_rate=progress_rate,
                    job_key=self.param_dict["job_key"]
                )

        return {"pred": results_pred, "proba": results_proba}

    def predict_decision(self, batch_x):
        rst: np.ndarray = self.model.predict(batch_x)
        if len(rst.shape) == 2 and rst.shape[1] > 1:
            rst = np.argmax(rst, axis=1)
        elif len(rst.shape) == 2 and rst.shape[1] == 1:
            rst = rst.flatten()

        return rst

    def predict_proba(self, batch_x):
        return self.model.predict_proba(batch_x)

    def load_model(self):
        raise NotImplementedError

    def saved_model(self):
        raise NotImplementedError

    def eval(self, dataset: dict):
        case: Callable = {
            "Classifier": self.eval_classifier,
            "Regressor": self.eval_regressor,
            "Clustering": self.eval_clustering,
            "WE": self.eval_we,
            "FE": self.eval_fe,
            "OD": self.eval_od,
            "TA": self.eval_ta,
        }.get(self.param_dict["algorithm_type"], None)
        try:
            return case(dataset=dataset)
        except Exception as e:
            self.LOGGER.error(e, exc_info=True)
            raise e

    def eval_classifier(self, dataset: dict):
        x = dataset["x"]
        _y = self._arg_max(dataset["y"])

        num_classes = self.param_dict["output_units"]

        pred = self.predict(x)["pred"]

        return self._eval_class_calculate(num_classes, _y, pred)

    @staticmethod
    def _eval_class_calculate(num_classes, _y, pred):
        results = list()

        for c in range(int(num_classes)):
            result = {
                "total": str(np.sum(np.equal(_y, c), dtype="int32")),
                # "TP": str(np.sum(np.take(np.equal(_y, c), np.where(np.equal(pred, c)))))  # 정탐
            }
            #####################
            for p in range(int(num_classes)):
                # actual: c, inference: p
                # if c = p, value is TP
                result[str(p)] = str(np.sum(np.take(np.equal(_y, c), np.where(np.equal(pred, p)))))
            #####################
            result["FN"] = str(int(result["total"]) - int(result[str(c)]))  # 미탐
            # self.AI_LOGGER.info(result)
            results.append(result)

        return results

    def eval_default_rst(self, dataset):
        x = dataset["x"]

        pred = self.predict_decision(x)
        try:
            pred = pred.tolist()
        except:
            pass

        return pred

    def eval_regressor(self, dataset: dict):
        return self.eval_default_rst(dataset)

    def eval_clustering(self, dataset: dict):
        return self.eval_default_rst(dataset)

    def eval_we(self, dataset: dict):
        return self.eval_default_rst(dataset)

    def eval_fe(self, dataset: dict):
        return self.eval_default_rst(dataset)

    def eval_od(self, dataset: dict):
        return self.eval_default_rst(dataset)

    def eval_ta(self, dataset: dict):
        return self.eval_default_rst(dataset)

    def _arg_max(self, y: np.ndarray) -> np.ndarray:
        try:
            # label encoding일 경우 shape가 (x, 1)이고 argmax 할 경우 전부 0으로 결과 반환
            y_shape = np.shape(y)
            if len(y_shape) == 2:
                if y_shape[1] >= 2:
                    _y = np.argmax(y, axis=1)
                else:
                    # 차원 축소
                    _y = np.squeeze(y, axis=1)
            else:
                _y = y
        except Exception as e:
            self.LOGGER.error(e, exc_info=True)
            _y = y

        return _y

