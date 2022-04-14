# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : manki.baek@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import joblib
from typing import Callable

from apeflow.common.Constants import Constants
from apeflow.interface.model.export.SavedModelAbstract import SavedModelAbstract


class SKLSavedModel(SavedModelAbstract):

    @classmethod
    def _save_case_fn(cls, model) -> Callable:
        return {
            Constants.OUT_MODEL_PKL: cls._save_model_pkl
        }.get(model.OUT_MODEL_TYPE, None)

    @classmethod
    def _save_model_pkl(cls, model, dir_model):
        joblib.dump(model.model, "{}/skl_model.joblib".format(dir_model))

    @classmethod
    def _load_case_fn(cls, model) -> Callable:
        return {
            Constants.OUT_MODEL_PKL: cls._load_model_pkl,
        }.get(model.OUT_MODEL_TYPE, None)

    @classmethod
    def _load_model_pkl(cls, model, dir_model):
        model.model = joblib.load("{}/skl_model.joblib".format(dir_model))
