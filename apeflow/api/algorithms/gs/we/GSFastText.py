# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from gensim.models.fasttext import FastText

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.gs.GSAlgAbstract import GSAlgAbstract


class GSFastText(GSAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "GSFastText"
    ALG_TYPE = ["WE"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_JSON

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(GSFastText, self).__init__(param_dict, wrapper, ext_data)

    def _build(self):
        self.first = True

    def learn(self, dataset):
        skip_window = self.param_dict["skip_window"]
        min_char_num = self.param_dict["min_char_num"]
        output_units = self.param_dict["output_units"]
        global_step = self.learn_params["global_step"]

        if self.first:
            dataset["x"] = self.remove_padding(dataset["x"])
            # data["x"]가 1D일 경우, 리스트에 있는 캐릭터 하나하나씩 학습함
            self.model = FastText(
                dataset["x"], window=skip_window, min_count=min_char_num,
                sg=0, iter=global_step, size=output_units, sorted_vocab=True
            )
            self.word_vector = self.model.wv
            self.first = False

    # def predict(self, x):
    #     predict_result = list()
    #     for row in x:
    #         row_result = list()
    #         for col in row:
    #             vec = self.model.wv[col].tolist()
    #             row_result.append(vec)
    #         predict_result.append(row_result)
    #
    #     return predict_result


if __name__ == '__main__':
    import numpy as np
    __dataset = {
        "x": np.array([["abc", "car", "#PADDING#"], ["apple", "#PADDING#", "#PADDING#"]])
    }

    __param_dict = {
        "algorithm_code": "GSFastText",
        "algorithm_type": "WE",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "2",
        "global_step": "1000",
        "model_nm": "GSFastText",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "job_key": "312513",
        "params": {
            "skip_window": "2",
            "min_char_num": "1",
        },

        "early_type": "0"
    }

    GSSG = GSFastText(__param_dict)
    GSSG._build()
    import gensim
    print("Gensim Version : {}".format(gensim.__version__))
    GSSG.learn(dataset=__dataset)
    print(GSSG.predict([["apple"]]))

    GSSG.saved_model()

    temp = GSFastText(__param_dict)
    temp.load_model()

    eval_data = {"x": [["abc"]]}
    print(temp.eval(dataset=eval_data))
