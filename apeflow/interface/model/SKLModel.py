# -*- coding: utf-8 -*-
# Author : Manki Baek
# e-mail : Manki.Baek@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from apeflow.interface.model.ModelAbstract import ModelAbstract


class SKLModel(ModelAbstract):
    def __init__(self, param_dict, ext_data=None):
        super(SKLModel, self).__init__(param_dict, ext_data)
        self.model = self._build()
