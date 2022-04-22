# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from gensim.models import Word2Vec

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.gs.GSAlgAbstract import GSAlgAbstract


class GSCBOW(GSAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "GSCBOW"
    ALG_TYPE = ["WE"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_JSON

    def __init__(self, param_dict, ext_data=None):
        super(GSCBOW, self).__init__(param_dict, ext_data)

    def _build(self):
        self.unknown_val = self.param_dict["unknown_val"]
        self.first = True

    def learn(self, dataset):
        super(GSCBOW, self).learn(dataset)
        skip_window = self.param_dict["skip_window"]
        min_char_num = self.param_dict["min_char_num"]
        output_units = self.param_dict["output_units"]
        global_step = self.learn_params["global_step"]

        if self.first:
            dataset["x"] = self.remove_padding(dataset["x"])

            # data["x"]가 1D일 경우, 리스트에 있는 캐릭터 하나하나씩 학습함
            self.model = Word2Vec(
                dataset["x"], window=skip_window, min_count=min_char_num,
                sg=0, iter=global_step, size=output_units, sorted_vocab=True,
                compute_loss=True, callbacks=[self.learn_result_callback]
            )

            self.word_vector = self.model.wv
            self.first = False


if __name__ == '__main__':
    __dataset = {
        "x": [["abc", "car", "#PADDING#"], ["apple", "#PADDING#", "#PADDING#"], ]
    }

    __param_dict = {
        "algorithm_code": "GSCBOW",
        "algorithm_type": "WE",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "3",
        "global_step": "10",
        "model_nm": "GSCBOW__1",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "job_key": "1315413",
        "params": {
            "skip_window": "2",
            "min_char_num": "1",
            "unknown_val": "4.0"
        },

        "early_type": "0"
    }

    GSSG = GSCBOW(__param_dict)
    GSSG._build()

    GSSG.learn(dataset=__dataset)
    print(GSSG.predict([["apple"]]))

    GSSG.saved_model()

    temp = GSCBOW(__param_dict)
    temp.load_model()

    eval_data = {"x": [["31131", "abc"], ["31131", "14gf"]]}
    print(temp.predict(eval_data))
