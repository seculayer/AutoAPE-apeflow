# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jin.kim@seculayer.com
# Powered by Seculayer © 2022 AI Service Model Team, R&D Center.
from typing import Dict, List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import numpy as np

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.skl.SKLAlgAbstract import SKLAlgAbstract
from pycmmn.exceptions.ParameterError import ParameterError
from pycmmn.rest.RestManager import RestManager


class SKLIsolationForest(SKLAlgAbstract):
    def learn_result_regressor(self, dataset):
        raise NotImplementedError

    # MODEL INFORMATION
    ALG_CODE = "SKLRandomForest"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(SKLIsolationForest, self).__init__(param_dict, wrapper, ext_data)

    def _check_parameter(self, param_dict):
        _param_dict = super(SKLIsolationForest, self)._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["n_estimators"] = int(param_dict.get("global_step", 20))
            _param_dict["contamination"] = float(param_dict.get("contamination", 0.1))
        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        self.model = IsolationForest(
            n_estimators=self.param_dict.get("n_estimators"),
            contamination=self.param_dict.get("contamination")
         )

    def predict_decision(self, batch_x):
        rst: np.ndarray = self.model.predict(batch_x)
        """
        predict()
            return :
                -1 = outlier
                 1 = inbound learning data(normal)
        """
        rst = np.where(rst == -1, 1, 0)

        return rst

    def predict_proba(self, batch_x):
        # results = list()
        # x = self.rtt_standard_scaler(x)
        score_val = self.model.decision_function(batch_x)
        # for d in score_val:
        #     res = d
        #     if d < 0.0:
        #         res = 1 + d
        #     if res > 1.0:
        #         res = 1.0
        #     results.append(res)
        score_val = np.where(score_val < -1, -1, score_val)
        score_val = np.where(score_val > 1, 1, score_val)
        score_val = np.subtract(score_val, 1)
        score_val = np.divide(score_val, 2)
        score_val = np.abs(score_val)

        return score_val

    def learn(self, dataset):
        train_dataset, valid_dataset = self._make_dataset(dataset)

        x = np.array(train_dataset.get("x"))
        self.model.set_params(
            max_samples=x.shape[0],
            max_features=x.shape[1],
        )
        self.model.fit(x)

        self.learn_result_isof(
            pred=self.predict(np.array(valid_dataset.get("x")))["pred"],
            label=valid_dataset.get("y")
        )

    def eval_classifier(self, dataset: dict):
        x = dataset["x"]
        _y = self._arg_max(dataset["y"])

        num_classes = self.param_dict["output_units"]

        pred = self.predict_decision(x)
        # pred = self.predict_proba(x)
        #
        # bin_pred = list()
        # for p in pred:
        #     v = 1
        #     if p < 0.5:
        #         v = 0
        #     bin_pred.append(v)

        return self._eval_class_calculate(num_classes, _y, pred)

    @staticmethod
    def rtt_standard_scaler(x):
        x = np.array(x)
        rtt = np.reshape(x[:, 3], (-1, 1))
        scaler = StandardScaler()
        rtt = scaler.fit_transform(rtt)

        not_rtt = x[:, :3]
        x = np.concatenate((not_rtt, rtt), axis=1)
        return x

    def _make_dataset(self, dataset) -> Tuple:
        train_dataset = {"x": list(), "y": list()}
        valid_dataset = {"x": list(), "y": list()}

        y = np.argmax(dataset.get("y"), axis=1)
        normal_length = 0
        for l in y:
            if l == 0:
                normal_length += 1

        normal_length = int(normal_length * 0.95)

        normal_idx = 0
        x = dataset["x"]
        for idx, label in enumerate(y):
            # if label == 1 or normal_idx >= normal_length:
            # #if label == 1:
            #     valid_dataset["x"].append(x[idx])
            #     valid_dataset["y"].append(label)
            # else:
            #     train_dataset["x"].append(x[idx])
            #     train_dataset["y"].append(label)
            #     normal_idx += 1
            valid_dataset["x"].append(x[idx])
            valid_dataset["y"].append(label)
            if label == 0 and normal_idx < normal_length:
                train_dataset["x"].append(x[idx])
                train_dataset["y"].append(label)
                normal_idx += 1
        # train_dataset["x"] = self.rtt_standard_scaler(train_dataset["x"])
        # valid_dataset["x"] = self.rtt_standard_scaler(valid_dataset["x"])

        return train_dataset,  valid_dataset

    def learn_result_isof(self, pred, label):
        pred_value = list()
        for idx, p in enumerate(pred):
            value = 0
            if p >= 0.5:
                value = 1
            pred_value.append([value])

        results = dict()
        results["global_sn"] = self.param_dict["global_sn"]
        results["accuracy"] = accuracy_score(y_pred=pred_value, y_true=label)
        loss = log_loss(y_true=label, y_pred=pred_value)
        results["loss"] = loss
        results["step"] = 1

        result_list = list()
        result_list.append(results)

        RestManager.update_learn_result(
            rest_url_root=Constants.REST_URL_ROOT,
            logger=self.LOGGER,
            job_key=self.param_dict["job_key"],
            rst=result_list
        )

        self.LOGGER.info(result_list)


if __name__ == '__main__':
    __dataset = {
        "x": np.array([[-12, -13], [-21, -15], [4, 21], [23, 13], [-12, -13], [-23, -12], [21, 11], [21, 11], [-21, -12], [-22, -12], [1, 12], [12, 21], [-31, -23], [-22, -15], [11, 11], [23, 11]]),
        # "y" : np.array([1, 1, 0, 0]),
        "y": np.array([[0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]]),
    }

    __param_dict = {
        "algorithm_code": "SKLRandomForest",
        "algorithm_type": "Classifier",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "global_step": "1000",
        "model_nm": "SKLISOFOREST-TEST",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "job_key": "3214523",
        "early_type": "0",
        "learning": "N",
        "params": {
            "n_estimators": 200
        }
    }
    iso = SKLIsolationForest(param_dict=__param_dict)
    iso.learn(__dataset)
    iso.saved_model()
    iso.load_model()
    pred = iso.predict(__dataset.get("x"))
    print(pred)

