# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : bmg8551@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import numpy as np

from gensim.models import Word2Vec

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.gs.GSAlgAbstract import GSAlgAbstract


class GSSkipGram(GSAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "GSSkipGram"
    ALG_TYPE = ["WE"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_JSON

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        super(GSSkipGram, self).__init__(param_dict, wrapper, ext_data)
        self.word_vector = None

    def _build(self):
        self.unknown_val = self.param_dict["unknown_val"]
        self.first = True

    def learn(self, dataset):
        super(GSSkipGram, self).learn(dataset)
        skip_window = self.param_dict["skip_window"]
        min_char_num = self.param_dict["min_char_num"]
        output_units = self.param_dict["output_units"]
        global_step = self.learn_params["global_step"]

        if self.first:
            x = self.remove_padding(dataset["x"])

            # data["x"]가 1D일 경우, 리스트에 있는 캐릭터 하나하나씩 학습함
            self.model = Word2Vec(
                x, window=skip_window, min_count=min_char_num,
                sg=1, iter=global_step, size=output_units, sorted_vocab=True,
                compute_loss=True, callbacks=[self.learn_result_callback]
            )

            self.word_vector = self.model.wv
            self.first = False


if __name__ == '__main__':
    import gensim
    print("gensim Version : {}".format(gensim.__version__))
    __dataset = {
        "x": np.array([["abc", "car", "mom"], ["apple", "auto", "#PADDING#"], ["apple", "auto", "#PADDING#"], ["apple", "car", "#PADDING#"], ["apple", "#PADDING#", "#PADDING#"] ])
    }
    print(__dataset["x"].shape)
    __param_dict = {
        "algorithm_code": "GSSkipGram",
        "algorithm_type": "WE",
        "data_type": "Single",
        "method_type": "Basic",
        "input_units": (2,),
        "output_units": "3",
        "model_nm": "GSSkipGram214",
        "alg_sn": "0",
        "job_type": "learn",
        "depth": "0",
        "global_sn": "0",
        "job_key": "124124124124",
        "global_step": "10",
        "params": {
            "skip_window": "2",
            "min_char_num": "1",
            "unknown_val": "3.0"
        },
        "early_type": "0"
    }

    GSSG = GSSkipGram(__param_dict)
    GSSG._build()

    GSSG.learn(dataset=__dataset)
    print(GSSG.predict([["apple"]]))

    GSSG.saved_model()

    temp = GSSkipGram(__param_dict)
    temp.load_model()

    eval_data = {"x": [["31131"]]}
    print(temp.eval(dataset=__dataset))
