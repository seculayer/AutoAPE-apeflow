# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import tensorflow as tf

from apeflow.common.Constants import Constants
from apeflow.common.Common import Common


class EarlyStopCallback(tf.keras.callbacks.Callback):
    def __init__(self, learn_params):
        super(EarlyStopCallback, self).__init__()

        self.learn_params = learn_params
        self.AI_LOGGER = Common.LOGGER.getLogger()

        self.prev_val = None
        self.early_steps = 0
        self.stopped_epoch = 0

    def stop_train(self, epoch):
        self.stopped_epoch = epoch
        self.model.stop_training = True
        self.AI_LOGGER.info("------ EARLY STOP !!!!! -----")

    def get_stopped_epoch(self):
        return self.stopped_epoch

    def on_epoch_end(self, epoch, logs=None):
        if not self.learn_params["early_type"] == Constants.EARLY_TYPE_NONE:
            key = self.learn_params["early_key"]

            curr_val = float(logs[key])

            if self.learn_params["early_type"] == Constants.EARLY_TYPE_MIN:
                if self.learn_params["minsteps"] < epoch:
                    if curr_val < self.learn_params["early_value"]:
                        self.stop_train(epoch)
                        return

            elif self.learn_params["early_type"] == Constants.EARLY_TYPE_MAX:
                if self.learn_params["minsteps"] < logs["step"]:
                    if curr_val > self.learn_params["early_value"]:
                        self.stop_train(epoch)
                        return

            elif self.learn_params["early_type"] == Constants.EARLY_TYPE_VAR:
                try:

                    if self.prev_val is not None:
                        if abs(curr_val - self.prev_val) < self.learn_params["early_value"]:
                            self.early_steps += 1
                        else:
                            self.early_steps = 0
                    self.prev_val = curr_val

                except:
                    pass

                if self.early_steps >= self.learn_params["minsteps"]:
                    self.stop_train(epoch)
                    return

        self.stopped_epoch = epoch
        return
