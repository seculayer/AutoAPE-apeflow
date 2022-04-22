# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import os
import json

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.AlgorithmAbstract import AlgorithmAbstract
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.interface.model.export.GSSavedModel import GSSavedModel
from apeflow.interface.utils.gs.LearnResultCallback import LearnResultCallback


class GSAlgAbstract(AlgorithmAbstract):
    # MODEL INFORMATION
    ALG_CODE = "GSAlgAbstract"
    ALG_TYPE = []
    DATA_TYPE = []
    VERSION = "1.0.0"
    LIB_TYPE = Constants.GENSIM
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_JSON

    def __init__(self, param_dict, ext_data=None):
        AlgorithmAbstract.__init__(self, param_dict, ext_data)
        self.learn_result_callback = None
        try:
            self.task_idx = int(json.loads(os.environ["TF_CONFIG"])["task"]["index"])
        except:
            self.task_idx = 0

        self.word_vector = None
        self.first = False
        self.unknown_val = None
        self.model = None
        self._build()
        self.index2word = None
        self.vectors = None
        self.counts = None

    @staticmethod
    def _check_parameter(param_dict):
        _param_dict = AlgorithmAbstract._check_parameter(param_dict)
        # Parameter Setting
        try:
            _param_dict["skip_window"] = int(param_dict["skip_window"])
            _param_dict["min_char_num"] = int(param_dict["min_char_num"])
            try:
                _param_dict["unknown_val"] = float(param_dict["unknown_val"])
            except:
                pass
        except:
            raise ParameterError
        return _param_dict

    def _build(self):
        raise NotImplementedError

    def learn(self, dataset):
        self.learn_result_callback = LearnResultCallback(
            job_key=self.param_dict["job_key"],
            data_len=len(dataset['x']),
            global_sn=self.param_dict["global_sn"]
        )

    def predict(self, x):
        output_units = self.param_dict["output_units"]

        predict_result = list()
        for row in x:
            row_result = list()
            for col in row:
                try:
                    vec = self.word_vector[col].tolist()
                except:
                    vec = [self.unknown_val] * output_units
                row_result.append(vec)
            predict_result.append(row_result)

        return predict_result

    def eval_we(self, dataset):
        if self.task_idx == 0:
            results = list()

            result = {"global_sn": self.param_dict["global_sn"]}
            word_list = self.index2word
            result["word_list"] = word_list
            try:
                result["vector_list"] = self.vectors.tolist()
            except:
                result["vector_list"] = self.vectors
            count_list = self.counts

            result["word_cnt_list"] = count_list

            results.append(result)

            return results
        else:
            return []

    def saved_model(self):
        GSSavedModel.save(model=self)

    def load_model(self):
        GSSavedModel.load(model=self)

    @staticmethod
    def remove_padding(x):
        try:
            _x = x.tolist()
        except:
            _x = x

        for row in _x:
            while True:
                try:
                    padding_idx = row.index("#PADDING#")
                    del(row[padding_idx])
                except:
                    break

        return _x
