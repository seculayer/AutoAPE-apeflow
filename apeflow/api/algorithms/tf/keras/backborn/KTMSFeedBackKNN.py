# -*- coding: utf-8 -*-
# Author : IlJu Mun
# e-mail : ilju.mun@seculayer.com
# Powered by Seculayer © 2021 AI Service Model Team, R&D Center.

# ---- python base packages
from typing import Dict

import numpy as np
import pickle

from apeflow.api.algorithms.tf.keras.backborn.KIDPSFeedBackKNNClassifier import KIDPSFeedBackKNNClassifier
from apeflow.api.algorithms.tf.keras.backborn.KTMSSiameseNetworkBackborn import KTMSSiameseNetworkBackborn
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract
from apeflow.common.Constants import Constants
from pycmmn.exceptions.ParameterError import ParameterError
from pycmmn.sftp.SFTPClientManager import SFTPClientManager
from pycmmn.utils.FileUtils import FileUtils

class KTMSFeedBackKNN(KIDPSFeedBackKNNClassifier):
    # MODEL INFORMATION
    ALG_CODE = "KTMSFeedBackKNN"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "2.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_IDPS_CLASSIFIER

    def __init__(self, param_dict, **kwargs):
        TFKerasAlgAbstract.__init__(self, param_dict, **kwargs)

        # back born variables
        self.bb_learning = kwargs.get("bb_learning", False)
        self.backborn = KTMSSiameseNetworkBackborn(param_dict, **kwargs)

        # knn variables
        self.clusters: Dict = None
        self.output_units = int(self.param_dict.get("output_units", 2))
        self.resource_model_path = f"{Constants.DIR_RESOURCES_MODEL}/tms/{self.VERSION}"
        self.info = None

        self.MRMS_SFTP_MANAGER: SFTPClientManager = SFTPClientManager(
            "{}:{}".format(Constants.MRMS_SVC, Constants.MRMS_SFTP_PORT),
            Constants.MRMS_USER, Constants.MRMS_PASSWD, self.LOGGER
        )

    def _check_parameter(self, param_dict):
        _param_dict = super(KTMSFeedBackKNN, self)._check_parameter(param_dict)
        try:
            _param_dict["r"] = float(param_dict.get("r", 0.001))
        except Exception:
            raise ParameterError
        return _param_dict

    def _build(self):
        pass

    def learn(self, data: Dict):
        if self.clusters is None:
            self._learning(data)
        return self._feedback_learning(data)

    def backborn_learning(self, data):
        self.backborn.learn(data, global_step=int(self.learn_params.get("global_step", 100)))
        self.backborn.backborn_saved_model()

    def eval_regressor(self, dataset):
        raise NotImplementedError

    def eval_we(self, dataset):
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    def eval_clustering(self, dataset):
        raise NotImplementedError

    def learn_result_regressor(self, data):
        raise NotImplementedError

    def _get_info(self):
        filename = f"{self.resource_model_path}/info/unique_info.pkl"
        with open(filename, "rb") as f:
            unique_info = pickle.load(f)
        return list(unique_info)

    def _make_init_dataset(self):
        self.info = self._get_info()
        labels = []
        for data in self.info:
            label = [0 for i in range(3)]
            label[int(data.get("label"))] = 1
            labels.append(label)
        data = {
            "x": np.load(f"{self.resource_model_path}/data/unique_features.npy") / 255.,
            "y": labels
        }
        return data

    def saved_model(self):
        dir_model = '{}/{}/{}'.format(
            Constants.DIR_TEMP, self.param_dict["model_nm"], self.param_dict["alg_sn"]
        )

        if not FileUtils.is_exist(dir_model):
            FileUtils.mkdir(dir_model)

        # back-born save
        self.backborn.model.save(dir_model + "/backborn.h5")
        self.backborn.model.save_weights(dir_model + "/backborn-weight.h5")

        # nearest feature save
        f = open("{}/clusters.model".format(dir_model), "wb")
        try:
            pickle.dump(self.clusters, f)
        except Exception as e:
            self.LOGGER.error(e, exc_info=True)
        finally:
            f.close()

        if self.backborn.task_idx == 0:
            self._scp_model_to_storage(dir_model, self.param_dict)

        self.LOGGER.info(f"{self.ALG_CODE} Saved...")
        self.LOGGER.info("model dir : {}".format(dir_model))

    def _scp_model_to_storage(self, dir_model: str, param_dict: dict) -> None:
        remote_path = f"{Constants.DIR_STORAGE}/{param_dict['model_nm']}"
        if not self.MRMS_SFTP_MANAGER.is_exist(remote_path):
            self.MRMS_SFTP_MANAGER.mkdir(remote_path)
        self.MRMS_SFTP_MANAGER.scp_to_storage(
            dir_model, remote_path
        )

if __name__ == '__main__':
    __params = {
        "algorithm_code": "KTMSFeedBackKNN",
        "algorithm_type": "Regressor",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (64,),
        "output_units": "3",
        "hidden_units": "256,128,64,32",
        "global_step": "10",
        "dropout_prob": "1.0",
        "optimizer_fn": "Adam",
        "model_nm": "TMS-SN-CLASSIFIER-2-0-0",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "learning_rate": "0.1",
        "params": {
            "r": 0.01,
        },
        "early_type": "2",
        "minsteps": "20",
        "early_key": "accuracy",
        "early_value": "0.99",

        "num_workers": "1",
        "job_key": "test",
        "learning": "Y"
    }
    TMS = KTMSFeedBackKNN(__params)
    data_set = TMS._make_init_dataset()
    result = TMS.learn(data_set) # learn 쪽에 global_step 받는걸 해결해여됨
    print(result)

    TMS.saved_model()
