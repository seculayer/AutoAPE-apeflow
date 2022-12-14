# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import json
import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Union

from apeflow.common.Constants import Constants
from apeflow.interface.model.export.TFSavedModel import TFSavedModel
from apeflow.api.algorithms.AlgorithmAbstract import AlgorithmAbstract
from apeflow.interface.utils.tf.keras.LearnResultCallback import LearnResultCallback
from apeflow.interface.utils.tf.keras.EarlyStopCallback import EarlyStopCallback
from pycmmn.rest.RestManager import RestManager


class TFKerasAlgAbstract(AlgorithmAbstract):
    # MODEL INFORMATION (STATIC)
    ALG_CODE = "TFKerasAlgAbstract"
    ALG_TYPE = []
    DATA_TYPE = []
    VERSION = "2.0.0"
    LIB_TYPE = Constants.KERAS
    TAG = "serve"
    OUT_MODEL_TYPE = Constants.OUT_MODEL_TF

    def __init__(self, param_dict, wrapper=None, ext_data=None):
        AlgorithmAbstract.__init__(self, param_dict, wrapper, ext_data=ext_data)

        self.num_workers = param_dict["num_workers"]

        self.input_name = "{}_{}_inputs".format(param_dict["model_nm"], param_dict["alg_sn"])
        self.output_name = "{}_{}_predicts".format(param_dict["model_nm"], param_dict["alg_sn"])

        # VARIABLES
        self.model = None

        try:
            self.task_idx = int(json.loads(os.environ["TF_CONFIG"])["task"]["index"])
        except:
            self.task_idx = 0

        if wrapper is None:
            # -- build model
            self._build()
        else:
            self.load_model()

    def _build(self):
        raise NotImplementedError

    def _make_train_dataset(self, data):
        len_data = len(data["x"])
        try:
            data["x"] = data["x"].astype(np.float32)
        except:
            pass
        buffer_size = len_data

        if buffer_size % self.batch_size != 0:
            parallel_step = buffer_size // self.batch_size + 1
        else:
            parallel_step = buffer_size // self.batch_size

        if "y" in data.keys():
            try:
                if "Regressor" in self.ALG_TYPE:
                    data["y"] = np.array(data["y"], dtype=np.float32)
                else:
                    data["y"] = np.array(data["y"])
            except:
                pass

            dataset = tf.data.Dataset.from_tensor_slices(
                (data["x"], data["y"])
            ).shuffle(
                buffer_size=self.batch_size
            ).repeat().batch(
                self.batch_size  # , drop_remainder=drop_remainder
            )

        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (data["x"])
            ).shuffle(
                buffer_size=self.batch_size
            ).repeat().batch(
                self.batch_size  # , drop_remainder=drop_remainder
            )

        # data shard option 해제
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        return dataset, parallel_step

    def saved_model(self):
        # if self.task_idx == 0:
        TFSavedModel.save(self)

    def load_model(self):
        TFSavedModel.load(self)

    def learn(self, dataset):
        # learn result
        global_step = self.learn_params["global_step"]
        global_sn = self.param_dict["global_sn"]

        l_dataset, v_dataset = self.split_data(dataset)

        result_callback = LearnResultCallback(
            global_sn=global_sn,
            job_key=self.param_dict["job_key"],
            epochs=global_step,
            task_idx=self.task_idx,
            data_len=len(l_dataset["x"])
        )

        v_dataset, v_parallel_step = self._make_train_dataset(v_dataset)
        l_dataset, l_parallel_step = self._make_train_dataset(l_dataset)

        # early stop
        early_stop_callback = EarlyStopCallback(
            learn_params=self.learn_params
        )
        self.model.fit(
            # x=l_dataset['x'], y=l_dataset['y'],
            x=l_dataset,
            steps_per_epoch=l_parallel_step,
            epochs=global_step,
            validation_data=v_dataset,
            validation_steps=v_parallel_step,
            callbacks=[result_callback, early_stop_callback],
            verbose=0,
        )

    def predict_decision(self, batch_x):
        batch_x = tf.keras.backend.cast(batch_x, tf.float32)
        return self.model(batch_x).numpy()

    def predict(self, x):
        batch_size = self.batch_size
        start = 0
        results_pred: Union[np.ndarray, None] = None
        results_proba: Union[np.ndarray, None] = None
        len_x = len(x)
        is_classifier: bool = True if self.param_dict["algorithm_type"] == "Classifier" else False

        while start < len_x:
            end = start + batch_size
            batch_x = x[start: end]
            if start == 0:
                results_pred = self.predict_decision(batch_x)
            else:
                results_pred = np.append(results_pred, self.predict_decision(batch_x), axis=0)
            start += batch_size

            if self.param_dict["learning"] == "N" and is_classifier:
                temp = len_x if start > len_x else start
                progress_rate = temp / len_x * 100
                RestManager.send_inference_progress(
                    rest_url_root=Constants.REST_URL_ROOT,
                    logger=self.LOGGER,
                    prograss_rate=progress_rate,
                    job_key=self.param_dict["job_key"]
                )

        if is_classifier:
            results_proba = results_pred
            results_pred = results_pred.argmax(axis=1)

        return {"pred": results_pred, "proba": results_proba}

    def split_data(self, dataset, l_ratio=80) -> Tuple[dict, dict]:
        l_dataset = dict()
        v_dataset = dict()
        len_data = len(dataset['x'])

        l_dataset['x'] = dataset['x'][: int(len_data * (l_ratio / 100))]
        v_dataset['x'] = dataset['x'][int(len_data * (l_ratio / 100)):]
        try:
            l_dataset['y'] = dataset['y'][: int(len_data * (l_ratio / 100))]
            v_dataset['y'] = dataset['y'][int(len_data * (l_ratio / 100)):]
        except Exception as e:
            l_dataset['y'] = None
            v_dataset['y'] = None
            self.LOGGER.error(str(e), exc_info=True)

        return l_dataset, v_dataset
