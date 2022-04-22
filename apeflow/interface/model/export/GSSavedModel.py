# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.
import os
import json
from typing import Callable

from pycmmn.utils.FileUtils import FileUtils
from gensim.models import KeyedVectors
from apeflow.common.Constants import Constants
from apeflow.interface.model.export.SavedModelAbstract import SavedModelAbstract


class GSSavedModel(SavedModelAbstract):

    @classmethod
    def _save_case_fn(cls, model) -> Callable:
        return {
            Constants.OUT_MODEL_JSON: cls._save_model_json
        }.get(model.OUT_MODEL_TYPE, None)

    @classmethod
    def _save_model_json(cls, model, dir_model):

        # saving for python eval process
        model.model.wv.save_word2vec_format("{}/word_vec.model".format(dir_model))

        _dict = model.model.wv.index2word
        _vec = model.model.wv.vectors
        _cnt = model.model.wv.vocab

        result_dict = dict()
        vocab_dict = dict()
        f = FileUtils.file_pointer("{}/apeflow_json.tmp".format(dir_model), "w")
        try:
            for idx, word in enumerate(_dict):
                temp_dict = {"vector": _vec[idx].tolist(), "index": idx, "cnt": _cnt["{}".format(word)].count}
                vocab_dict[word] = temp_dict
            result_dict["ALG_CODE"] = model.ALG_CODE
            result_dict["ALG_TYPE"] = model.param_dict["algorithm_type"]
            result_dict["vocab_dict"] = vocab_dict
            f.write(
                json.dumps(result_dict, indent=4)
            )
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e
        finally:
            f.close()

        os.rename("{}/apeflow_json.tmp".format(dir_model), "{}/apeflow_json.model".format(dir_model))

    @classmethod
    def _load_case_fn(cls, model) -> Callable:
        return {
            Constants.OUT_MODEL_JSON: cls._load_model_json
        }.get(model.OUT_MODEL_TYPE, None)

    @classmethod
    def _load_model_json(cls, model, dir_model):
        model.word_vector = KeyedVectors.load_word2vec_format("{}/word_vec.model".format(dir_model), binary=False)

        f = FileUtils.file_pointer("{}/apeflow_json.model".format(dir_model), "r")
        try:
            dictionary = json.load(f)
            dict_len = len(dictionary["vocab_dict"])

            model.index2word = [-1] * dict_len
            model.vectors = [-1] * dict_len
            model.counts = [-1] * dict_len

            key_list = list(dictionary["vocab_dict"].keys())
            for word in key_list:
                idx = dictionary["vocab_dict"][word]["index"]
                vector = dictionary["vocab_dict"][word]["vector"]
                cnt = dictionary["vocab_dict"][word]["cnt"]
                model.index2word[idx] = word
                model.vectors[idx] = vector
                model.counts[idx] = cnt
        except Exception as e:
            cls.LOGGER.error(e, exc_info=True)
            raise e
        finally:
            f.close()
