# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : Manki.Baek@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.


from apeflow.interface.model.ModelAbstract import ModelAbstract
from apeflow.api.algorithms.gs.GSAlgAbstract import GSAlgAbstract


class GSModel(ModelAbstract):
    def __init__(self, param_dict, ext_data=None):
        ModelAbstract.__init__(self, param_dict, ext_data)
        self.model: GSAlgAbstract = self._build()

    def learn(self, dataset):
        self.model.learn(dataset)
        self.model.saved_model()
