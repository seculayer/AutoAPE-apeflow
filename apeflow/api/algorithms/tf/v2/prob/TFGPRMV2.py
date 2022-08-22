# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2019 Intelligence Team, R&D Center.

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import numpy as np
import tensorflow_probability as tfp
import json
import os

from apeflow.common.Constants import Constants
from apeflow.api.algorithms.tf.v2.TFV2AlgAbstract import TFV2AlgAbstract
from pycmmn.rest.RestManager import RestManager


# class : class_name
class TFGPRMV2(TFV2AlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "TFGPRMV2"
    OUT_MODEL_TYPE = Constants.OUT_MODEL_JSON
    ALG_TYPE = ["TA"]

    def __init__(self, **kwargs):
        super(TFGPRMV2, self).__init__(**kwargs)
        self.model = dict()

    def _check_parameter(self, param_dict):
        _param_dict = super()._check_parameter(param_dict)

        # Algorithm PARAMS
        _param_dict["seq_length"] = self._decision_time_sequence(
            param_dict["seq_type"], int(param_dict["seq_term"])
        )
        if param_dict.get("use_lower", "False").lower() == "false":
            _param_dict["use_lower"] = False
        else:
            _param_dict["use_lower"] = True

        # Learning PARAMS
        _param_dict["learning_rate"] = float(param_dict["learning_rate"])

        return _param_dict

    def _decision_time_sequence(self, seq_type, seq_term=1):
        hour = 24
        term = int(60 / seq_term)
        day = term * hour
        if seq_type == "day":
            return day
        elif seq_type == "week":
            return day * 7
        elif seq_type == "month":
            return day * 30

        return day

    def _build(self):
        # get Model PARAMS
        model_nm = "{}_{}".format(self.param_dict["model_nm"], self.param_dict["alg_sn"])

        # setting vars
        self.amplitude = tf.Variable(np.float64(0), name='{}_amplitude'.format(model_nm))
        self.length_scale = tf.Variable(np.float64(0), name='{}_length_scale'.format(model_nm))

        self.observation_noise_variance = tf.Variable(np.float64(-5), name='{}_observation_noise_variance'.format(model_nm))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.param_dict["learning_rate"])
        self.trainable_var_list = [self.amplitude, self.length_scale, self.observation_noise_variance]

    def create_kernel(self):
        amplitude = tf.exp(self.amplitude)
        length_scale = tf.exp(self.length_scale)
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

        return kernel

    # @tf.function(autograph=False, jit_compile=True)
    def loss_fn(self, batch_x, batch_y):

        observation_noise_variance = tf.exp(self.observation_noise_variance)
        observation_index_points = tf.expand_dims(batch_x, -1)

        gp = tfp.distributions.GaussianProcess(
            kernel=self.create_kernel(),
            index_points=observation_index_points,
            observation_noise_variance=observation_noise_variance
        )
        log_probs = gp.log_prob(batch_y)
        return -tf.reduce_mean(log_probs)

    def learn(self, dataset):
        _X = dataset["x"].reshape(-1)
        _Y = dataset["y"].reshape(-1)

        learn_result_list = list()

        # tensor abstract learn()
        for epoch in range(self.learn_params["global_step"]):
            # gaussian learn()
            try:
                BATCH_SIZE = self.batch_size
                len_data_x = len(dataset["x"])
                n_batch = int(len_data_x / BATCH_SIZE)
                if len_data_x % BATCH_SIZE != 0:
                    n_batch += 1
                total_nll = 0
                start = 0
                end = BATCH_SIZE
                for i in range(n_batch):
                    with tf.GradientTape() as tape:
                        l = self.loss_fn(_X[start:end], _Y[start:end])
                    g = tape.gradient(l, self.trainable_var_list)
                    self.optimizer.apply_gradients(zip(g, self.trainable_var_list))

                    total_nll += l
                    start += BATCH_SIZE
                    end += BATCH_SIZE

                self.nll = total_nll / n_batch

                self.data_x = _X
                self.data_y = _Y
            except Exception as e:
                self.LOGGER.error(e, exc_info=True)
                raise Exception

            learn_result_list.append(self.learn_result(epoch))

            if json.loads(os.environ["TF_CONFIG"])["task"]["index"] == "0":
                RestManager.update_learn_result(
                    rest_url_root=Constants.REST_URL_ROOT,
                    logger=self.LOGGER,
                    job_key=self.param_dict["model_nm"],
                    rst=learn_result_list
                )

        return learn_result_list

    def learn_result(self, epoch):
        results = dict()
        results["step"] = str(epoch)
        results["NLL"] = str(self.nll.numpy())

        self.LOGGER.info(results)

        return results

    def saved_model(self):
        BATCH_SIZE = self.batch_size
        seq_length = self.param_dict["seq_length"]
        len_data_x = len(self.data_x)
        n_batch = int(len_data_x / BATCH_SIZE)
        if len_data_x % BATCH_SIZE != 0:
            n_batch += 1
        start = 0
        end = BATCH_SIZE
        mean = None
        f_var = None
        # FOR GPRM
        index_points = tf.expand_dims(
            tf.cast(
                tf.linspace(
                    1.0, float(seq_length),
                    seq_length
                ),
                dtype=tf.float64
            ),
            -1
        )
        observation_noise_variance = tf.exp(self.observation_noise_variance)

        for i in range(n_batch):
            observation_index_points = tf.expand_dims(self.data_x[start: end], -1)

            gprm = tfp.distributions.GaussianProcessRegressionModel(
                kernel=self.create_kernel(),
                index_points=index_points,
                observation_index_points=observation_index_points,
                observations=self.data_y[start:end],
                observation_noise_variance=observation_noise_variance,
            )
            _mean = gprm.mean()
            _f_var = gprm.stddev()

            if i != 0:
                mean = (mean + _mean)/2
                f_var = (f_var + _f_var) / 2
            else:
                mean = _mean
                f_var = _f_var

            start += BATCH_SIZE
            end += BATCH_SIZE
            if end >= len_data_x:
                end = -1
            self.LOGGER.debug("batch : {} / {}".format(i + 1, n_batch))

        # self.LOGGER.info(f"mean : {mean}, stddev : {f_var}")

        upper_bound = mean + 1.96 * f_var
        lower_bound = mean - 1.96 * f_var

        for i, _ip in enumerate(index_points):
            result = dict()
            result["upper_bound"] = "{}".format(upper_bound[i])
            if not self.param_dict.get("use_lower"):
                result["lower_bound"] = "{}".format(0)
            else:
                result["lower_bound"] = "{}".format(lower_bound[i])
            result["mean"] = "{}".format(mean[i])
            self.model[int(_ip)] = result

        TFV2AlgAbstract.saved_model(self)

    def eval_ta(self, dataset):
        _X = dataset["x"]
        _Y = dataset["y"]

        results = list()

        for i in range(len(_X)):
            result = dict()
            result["eval_type"] = "1"
            result["global_sn"] = self.param_dict["global_sn"]
            result["feature_x"] = "{}".format(int(_X[i][0]))
            result["feature_y"] = "{}".format(_Y[i][0])
            results.append(result)

        for _key in self.model.keys():
            if _key == "ALG_CODE" or _key == "ALG_TYPE" :
                continue
            result = dict()
            result["eval_type"] = "2"
            result["global_sn"] = self.param_dict["global_sn"]
            result["feature_x"] = "{}".format(int(_key))
            result["upper_bound"] = "{}".format(float(self.model[_key]["upper_bound"]))
            result["lower_bound"] = "{}".format(float(self.model[_key]["lower_bound"]))
            result["mean"] = "{}".format(self.model[_key]["mean"])
            results.append(result)
        # self.LOGGER.info(results)
        return results

    def predict(self, dataset):
        """
        :param dataset: Dict. data of other model type is list, but it is excepted.
                        branch code : ModelInterface.py
        :return:
        """
        results = list()
        for idx, x in enumerate(dataset["x"]):
            rst_list = list()
            try:
                ranges = self.model[str(int(x[0]))]
            except:
                ranges = self.model[str(x[0])]

            y = dataset["y"][idx][0]
            if float(ranges["lower_bound"]) <= y <= float(ranges["upper_bound"]):
                rst_list.append(0)
            else:
                rst_list.append(1)

            rst_list.append(x[0])
            rst_list.append(y)
            results.append(rst_list)
        return results


if __name__ == '__main__':
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # print("physical devices: ", physical_devices)
    # for gpu_no in range(4):
    #     tf.config.experimental.set_memory_growth(physical_devices[gpu_no], True)
    param_dict = {
        "params": {
            # ALGORITHM PARAMS
            "algorithm_code": "TFGPRMV2",
            "algorithm_type": "TA",
            "data_type": "Single",
            "method_type": "Basic",
            "input_units": "1",
            "output_units": "1",
            "seq_type": "day",         # day, week, month
            "seq_term": "1",
            # "num_samples" : "0",
            # "predict_ratio" : "0.0",

            # MODEL PARAMS
            "model_nm": "test_1",
            "alg_sn": "0",
            "job_type": "learn",
            "depth": "0",
            "global_sn": "0",

            # LERANING PARAMS
            "global_step": "200",
            "learning_rate": "0.1",
            "early_type": "3",
            "minsteps": "10",
            "early_key": "NLL",
            "early_value": "0.1",
            "job_key": "1",
            "learning": "Y"
        },
        "num_workers": "1",
        "model_nm": "test_1",
        "alg_sn": "0",
    }

    gprm = TFGPRMV2(param_dict=param_dict, ext_data=None)

    repeat = 1
    seq_length = gprm.param_dict["seq_length"]
    samples = seq_length * repeat
    # x = np.random.randint(minute*hour+1, size=(samples, 1))
    x_list = []
    for _ in range(repeat):
        for j in range(seq_length):
            x_list.append([float(1 + j)])

    x = np.array(x_list)
    y = np.random.random((samples, 1))

    for i, _y in enumerate(y):
        _y[0] = x[i][0] % 25 + int(_y[0]*5) + 5

    data = {
        "x": x,
        "y": y
    }

    gprm._build()

    # # #
    gprm.learn(dataset=data)

    gprm.saved_model()

    tem = TFGPRMV2(param_dict=param_dict, ext_data=None)
    tem.load_model()

    tem.eval(data)
    results = tem.predict(data)

    print(results)

    print("end")
