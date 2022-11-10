# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.
import os
import json
import joblib
from typing import Callable
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from pycmmn.utils.FileUtils import FileUtils
from apeflow.common.Constants import Constants
from apeflow.interface.model.export.SavedModelAbstract import SavedModelAbstract


class TFSavedModel(SavedModelAbstract):

    @classmethod
    def save(cls, model):
        if model.task_idx != 0:
            return
        super(TFSavedModel, cls).save(model)

    @classmethod
    def _save_case_fn(cls, model) -> Callable:
        return {
            Constants.OUT_MODEL_TF: cls._save_model_keras if model.LIB_TYPE == Constants.KERAS else None,
            Constants.OUT_MODEL_XGB: cls._save_model_xgb,
            Constants.OUT_MODEL_JSON: cls._save_model_json,
            Constants.OUT_MODEL_KERAS_TOKENIZER: cls._save_model_keras_tokenizer,
            Constants.OUT_MODEL_APE_OUTLIER_DETCTION: cls._save_model_ape_outlier_detection
        }.get(model.OUT_MODEL_TYPE, None)

    @classmethod
    def _save_model_keras(cls, model, dir_model):
        # model.model.save_weights(dir_model + '/weights.h5', save_format='h5')
        model.model.save(dir_model)

    @classmethod
    def _save_model_xgb(cls, model, dir_model):
        FileUtils.mkdir(dir_model)
        model.model.save_model(dir_model + "/model.h5")

    @classmethod
    def _save_model_json(cls, model, dir_model):
        FileUtils.mkdir(dir_model)
        model_dict = model.model
        model_dict["ALG_CODE"] = model.ALG_CODE
        model_dict["ALG_TYPE"] = model.param_dict["algorithm_type"]
        model_nm = "{}/apeflow_json".format(dir_model)
        f = FileUtils.file_pointer("{}.tmp".format(model_nm), "w")
        try:
            json.dump(model_dict, f, indent=4)
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e
        finally:
            f.close()
        os.rename("{}.tmp".format(model_nm), "{}.model".format(model_nm))

    @classmethod
    def _save_model_keras_tokenizer(cls, model, dir_model):
        FileUtils.mkdir(dir_model)
        model_nm = "{}/apeflow_json".format(dir_model)
        f = FileUtils.file_pointer("{}.tmp".format(model_nm), "w")
        try:
            f.write(model.model.to_json())
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e
        finally:
            f.close()
        os.rename("{}.tmp".format(model_nm), "{}.model".format(model_nm))

    @classmethod
    def _save_model_ape_outlier_detection(cls, model, dir_model):
        cls._save_model_keras(model.model, dir_model)
        model_nm = "{}/outlier_kmeans".format(dir_model)
        joblib.dump(model.kmeans_model, "{}.pkl".format(model_nm))
        result_dict = dict()
        result_dict["ALG_CODE"] = model.ALG_CODE
        result_dict["ALG_TYPE"] = model.param_dict["algorithm_type"]
        result_dict["max_distances"] = model.max_distances

        f = FileUtils.file_pointer("{}.tmp".format(model_nm), "w")
        try:
            json.dump(result_dict, f, indent=4)
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e
        finally:
            f.close()
        os.rename("{}.tmp".format(model_nm), "{}.json".format(model_nm))

    @classmethod
    def _load_case_fn(cls, model) -> Callable:
        return {
            Constants.OUT_MODEL_TF: cls._load_model_keras if model.LIB_TYPE == Constants.KERAS else None,
            Constants.OUT_MODEL_XGB: cls._load_model_xgb,
            Constants.OUT_MODEL_JSON: cls._load_model_json,
            Constants.OUT_MODEL_KERAS_TOKENIZER: cls._load_model_keras_tokenizer,
            Constants.OUT_MODEL_APE_OUTLIER_DETCTION: cls._load_model_ape_outlier_detection
        }.get(model.OUT_MODEL_TYPE, None)

    @classmethod
    def _load_model_keras(cls, model, dir_model):
        try:
            # model.model.load_weights(dir_model + '/weights.h5')
            model.model = tf.keras.models.load_model(dir_model)
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e

    @classmethod
    def _load_model_xgb(cls, model, dir_model):
        try:
            # model.model.load_weights(dir_model + '/weights.h5')
            model.model.load_model(dir_model + "/model.h5")
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e

    @classmethod
    def _load_model_json(cls, model, dir_model):
        model_nm = "{}/apeflow_json".format(dir_model)
        f = FileUtils.file_pointer("{}.model".format(model_nm), "r")
        try:
            model.model = json.load(f)
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e
        finally:
            f.close()

    @classmethod
    def _load_model_keras_tokenizer(cls, model, dir_model):
        model_nm = "{}/apeflow_json".format(dir_model)
        f = FileUtils.file_pointer("{}.model".format(model_nm), "r")
        try:
            data = f.read()
            model.model = tokenizer_from_json(data)
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e
        finally:
            f.close()

    @classmethod
    def _load_model_ape_outlier_detection(cls, model, dir_model):
        model_nm = "{}/outlier_kmeans".format(dir_model)
        cls._load_model_keras(model.model, dir_model)
        model.kmeans_model = joblib.load("{}.pkl".format(model_nm))

        f = FileUtils.file_pointer("{}.json".format(model_nm), "r")
        try:
            model.max_distances = json.load(f).get("max_distances")
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e
        finally:
            f.close()
