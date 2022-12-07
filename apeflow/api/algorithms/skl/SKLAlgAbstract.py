# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : manki.baek@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from sklearn.metrics import log_loss
import numpy as np

from pycmmn.rest.RestManager import RestManager
from apeflow.common.Constants import Constants
from apeflow.api.algorithms.AlgorithmAbstract import AlgorithmAbstract
from apeflow.interface.model.export.SKLSavedModel import SKLSavedModel


class SKLAlgAbstract(AlgorithmAbstract):
    # MODEL INFORMATION
    ALG_CODE = "SKLAlgAbstract"
    ALG_TYPE = []
    DATA_TYPE = []
    VERSION = "1.0.0"
    LIB_TYPE = Constants.SCIKIT_LEARN
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_PKL

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(SKLAlgAbstract, self).__init__(param_dict, wrapper, ext_data)
        # VARIABLES
        self.model = None

        if wrapper is None:
            self._build()
        else:
            self.load_model()

    def _build(self):
        raise NotImplementedError

    def learn(self, dataset):
        raise NotImplementedError

    def learn_result(self, dataset):
        if self.param_dict["algorithm_type"] == "Classifier":
            results = self.learn_result_classifier(dataset=dataset)
        elif self.param_dict["algorithm_type"] == "Regressor":
            results = self.learn_result_regressor(dataset=dataset)
        elif self.param_dict["algorithm_type"] == "Clustering":
            results = self.learn_result_clustering(dataset=dataset)
        elif self.param_dict["algorithm_type"] == "FE":
            results = self.learn_result_fe(dataset=dataset)
        elif self.param_dict["algorithm_type"] == "OD":
            results = self.learn_result_od(dataset=dataset)
        else:
            raise NotImplementedError

        RestManager.update_learn_result(
            rest_url_root=Constants.REST_URL_ROOT,
            logger=self.LOGGER,
            job_key=self.param_dict["job_key"],
            rst=results
        )

        self.LOGGER.info(results)
        return results

    def learn_result_classifier(self, dataset):
        results = dict()
        # results["global_sn"] = self.param_dict["global_sn"]
        results["accuracy"] = self.model.score(X=dataset["x"], y=self._arg_max(dataset["y"]))
        loss = log_loss(self._arg_max(dataset["y"]), self.predict(dataset["x"])["pred"])
        results["loss"] = loss
        results["step"] = self.learn_params.get("global_step", 1)

        result_list = list()
        result_list.append(results)

        return result_list

    def learn_result_regressor(self, dataset):
        raise NotImplementedError

    def learn_result_clustering(self, dataset):
        results = dict()
        results["global_sn"] = self.param_dict["global_sn"]
        results["step"] = -1
        # loss = log_loss(data["y"], self.predict(dataset["x"]))
        # results["loss"] = loss
        # results["label"] = self._model.fit_predict(dataset["x"])
        result_list = list()
        result_list.append(results)

        return result_list

    def learn_result_fe(self, dataset):
        results = dict()
        results["global_sn"] = self.param_dict["global_sn"]
        results["step"] = -1
        # loss = log_loss(data["y"], self.predict(dataset["x"]))
        # results["loss"] = loss

        result_list = list()
        result_list.append(results)

        return result_list

    def learn_result_od(self, dataset):
        results = dict()
        results["global_sn"] = self.param_dict["global_sn"]
        results["step"] = -1
        # loss = log_loss(data["y"], self.predict(data))
        # results["loss"] = loss
        # results["label"] = self._model.fit_predict(data["x"])
        result_list = list()
        result_list.append(results)

        return result_list

    def saved_model(self):
        SKLSavedModel.save(model=self)

    def load_model(self):
        SKLSavedModel.load(model=self)

    def eval_clustering(self, dataset):
        x = dataset["x"]

        result_dict = dict()
        result_dict["global_sn"] = self.param_dict["global_sn"]

        try:
            result_dict["cluster"] = self.predict_decision(x).tolist()
            result_dict["features"] = x.tolist()
        except:
            result_dict["cluster"] = self.predict_decision(x)
            result_dict["features"] = x

        result_list = list()
        result_list.append(result_dict)

        return result_list

    def eval_fe(self, dataset):
        x = dataset["x"]

        result_list = list()

        result_dict = dict()
        result_dict["global_sn"] = self.param_dict["global_sn"]
        try:
            result_dict["predicts"] = self.predict_decision(x).tolist()
        except:
            result_dict["predicts"] = self.predict_decision(x)
        result_dict["cost"] = "None"

        result_list.append(result_dict)
        return result_list

    def eval_od(self, dataset):
        x = dataset["x"]

        result_dict = dict()
        result_dict["global_sn"] = self.param_dict["global_sn"]

        try:
            result_dict["predicts"] = self.predict_decision(x).tolist()
        except:
            result_dict["predicts"] = self.predict_decision(x)

        result_list = list()
        result_list.append(result_dict)

        return result_list
