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
        loss = log_loss(self._arg_max(dataset["y"]), self.predict(dataset["x"]))
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

    def predict(self, x):
        batch_size = self.batch_size
        start = 0
        results = list()
        len_x = len(x)

        while start < len_x:
            end = start + batch_size
            batch_x = x[start: end]
            try:
                results.extend(self.model.predict(batch_x).tolist())
            except:
                results.extend(self.model.predict(batch_x))
            start += batch_size

            if self.param_dict["learning"] == "N":
                progress_rate = start / len_x * 100
                RestManager.send_inference_progress(
                    rest_url_root=Constants.REST_URL_ROOT,
                    logger=self.LOGGER,
                    prograss_rate=progress_rate,
                    job_key=self.param_dict["job_key"]
                )

        results = np.array(results)
        if self.param_dict["algorithm_type"] == "Classifier":
            if len(results.shape) >= 2 and results.shape[1] >= 2:  # case onehot encoding
                results = np.argmax(results, axis=1)
        return results

    def saved_model(self):
        SKLSavedModel.save(model=self)

    def load_model(self):
        SKLSavedModel.load(model=self)

    def eval_classifier(self, dataset: dict):
        x = dataset["x"]
        _y = self._arg_max(dataset["y"])

        num_classes = self.param_dict["output_units"]

        pred = self.predict(x)

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

    def eval_clustering(self, dataset):
        x = dataset["x"]

        result_dict = dict()
        result_dict["global_sn"] = self.param_dict["global_sn"]

        try:
            result_dict["cluster"] = self.predict(x).tolist()
            result_dict["features"] = x.tolist()
        except:
            result_dict["cluster"] = self.predict(x)
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
            result_dict["predicts"] = self.predict(x).tolist()
        except:
            result_dict["predicts"] = self.predict(x)
        result_dict["cost"] = "None"

        result_list.append(result_dict)
        return result_list

    def eval_od(self, dataset):
        x = dataset["x"]

        result_dict = dict()
        result_dict["global_sn"] = self.param_dict["global_sn"]

        try:
            result_dict["predicts"] = self.predict(x).tolist()
        except:
            result_dict["predicts"] = self.predict(x)

        result_list = list()
        result_list.append(result_dict)

        return result_list

    @staticmethod
    def _arg_max(y: list) -> list:
        try:
            _y = np.argmax(y, axis=1).tolist()
        except:
            _y = y

        return _y
