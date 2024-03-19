# -*- coding: utf-8 -*-
# Author : IlJu Mun
# e-mail : ilju.mun@seculayer.com
# Powered by Seculayer © 2021 AI Service Model Team, R&D Center.

# ---- python base packages
import math
import pickle
from typing import Dict, List
import os

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import BallTree

from apeflow.api.algorithms.tf.keras.backborn.KIDPSCNNBackborn import KIDPSCNNBackborn
from apeflow.api.algorithms.tf.keras.TFKerasAlgAbstract import TFKerasAlgAbstract
from apeflow.common.Constants import Constants
from pycmmn.exceptions.ParameterError import ParameterError
from apeflow.interface.utils.tf.TFUtils import TFUtils
from pycmmn.utils.FileUtils import FileUtils


class KIDPSFeedBackKNNClassifier(TFKerasAlgAbstract):
    # MODEL INFORMATION
    ALG_CODE = "KIDPSFeedBackKNNClassifier"
    ALG_TYPE = ["Classifier"]
    DATA_TYPE = ["Single"]
    VERSION = "1.0.0"
    DIST_TYPE = Constants.DIST_TYPE_SINGLE
    OUT_MODEL_TYPE = Constants.OUT_MODEL_IDPS_CLASSIFIER

    def __init__(self, param_dict, **kwargs):
        super().__init__(param_dict, **kwargs)

        # back born variables
        self.bb_learning = kwargs.get("bb_learning", False)
        self.backborn = KIDPSCNNBackborn(param_dict, **kwargs)

        # knn variables
        self.clusters: Dict = None
        self.output_units = int(self.param_dict.get("output_units", 2))

    def _check_parameter(self, param_dict):
        _param_dict = super(KIDPSFeedBackKNNClassifier, self)._check_parameter(param_dict)
        try:
            _param_dict["r"] = float(param_dict.get("r", 0.001))
        except Exception:
            raise ParameterError
        return _param_dict

    def _build(self):
        pass

    def backborn_learning(self, data):
        self.backborn.learn(data, global_step=int(self.learn_params.get("global_step", 100)))
        self.backborn.backborn_saved_model()

    def learn(self, data: Dict, global_step: int):
        if self.clusters is None:
            return self._learning(data, global_step)
        else:
            return self._feedback_learning(data)

    def predict(self, x):
        results = list()
        extracted_features: Dict = self.backborn.predict(x)
        cluster_pred = self.clusters.get("centroid_knn").query(extracted_features.get("pred"), k=1, return_distance=False)
        for idx, c in enumerate(cluster_pred):
            results.append(
                self._knn_classifier(
                    extracted_features.get("pred")[idx],
                    self.clusters.get("clusters")[c[0]]
                )
            )

        results = np.array(results)
        results_pred = results.argmax(axis=1)

        return {"pred" : results_pred, "proba" : results}

    def _learning(self, data):
        results = None
        # backborn learning
        if not FileUtils.is_exist(
            '{}/{}/{}'.format(
                Constants.DIR_LOAD_MODEL, self.param_dict["model_nm"],
                self.param_dict["alg_sn"]
            ) + "/backborn.h5"
        ) and self.bb_learning:
            results = self.backborn.learn(data)
        return self._init_cluster(data, results)

    def _init_cluster(self, data, results):
        # 2022.09.23 추가 Sampling
        sample_size = int(len(data.get("x")) / 16) + 1
        if sample_size >= 20000:
            sample_size = 20000
        cluster_sample_x = data.get("x")[:sample_size]
        cluster_sample_y = data.get("y")[:sample_size]

        if self.clusters is None:
            # data.get("x") -> cluster_sample_x
            extracted_features = self.backborn.predict(cluster_sample_x).get("pred")
            # data.get("y") -> cluster_sample_y
            labels = cluster_sample_y

            leaf_size = int(math.log2(len(cluster_sample_x) / 16))
            if leaf_size < 2:
                leaf_size = 2
            ball_tree = BallTree(extracted_features, leaf_size=leaf_size)
            pred = ball_tree.query_radius(
                extracted_features,
                r=self.param_dict["r"], count_only=False, return_distance=False)

            # make cluster
            self.clusters = self._make_cluster(pred, extracted_features, labels)
            self.print_cluster_statistics(self.clusters.get("clusters"))

            # centroid knn
            centroid = self.clusters.get("centroid")
            # data.get("x") -> cluster_sample_x
            leaf_size = int(math.log2(len(cluster_sample_x) / 16))
            if leaf_size < 2:
                leaf_size = 2
            self.clusters["centroid_knn"] = BallTree(np.array(centroid), leaf_size=leaf_size)
        return results

    def _feedback_learning(self, data):
        self.backborn.backborn_load_model()
        extracted_features = self.backborn.predict(data.get("x")).get("pred")
        labels = data.get("y")

        steps = 0
        model_results: List[Dict] = list()
        feedback = True
        confusion = None
        while feedback:
            feedback = False
            results = list()
            total = 0
            tp = 0

            cluster_pred = self.clusters.get("centroid_knn").query(extracted_features, k=1, return_distance=False)
            for idx, c in enumerate(cluster_pred):
                # KNN
                label = self._get_label(labels[idx])
                feature = extracted_features[idx]
                pred = np.argmax(self._knn_classifier(feature, self.clusters.get("clusters")[c[0]]))

                # feedback
                if int(pred) != int(label):
                    feedback = True
                    self._feedback(c[0], feature, label)
                else:
                    tp += 1
                results.append(pred)
                total += 1

                if idx % 1000 == 0:
                    self.LOGGER.info("feedback process ... : {}".format(idx))

            model_results.append(
                {"step": steps, "accuracy": round(float(tp)/total, 4),
                 "global_sn": self.param_dict.get("global_sn", "0")}
            )
            feedback, confusion = self._check_changed(feedback, confusion, labels, results)
            steps += 1
            # info logs
            TFUtils.print_confusion_matrix(results, labels, confusion)
            if tp == total:
                break
        self.print_cluster_statistics(self.clusters.get("clusters"))

        return model_results

    def eval_classifier(self, dataset):
        self._feedback_learning(dataset)
        rst = super(KIDPSFeedBackKNNClassifier, self).eval_classifier(dataset)

        return rst

    def _check_changed(self, feedback, confusion, labels, pred):
        changed = False

        # curr_confusion = confusion_matrix(y_true=y_true, y_pred=pred)
        curr_confusion = self._calc_confusion_matrix(labels, pred)

        if confusion is None:
            return feedback, curr_confusion

        if len(confusion) == len(curr_confusion) and len(confusion[0]) == len(curr_confusion[0]):
            for i, row in enumerate(confusion):
                for j, con in enumerate(row):
                    if confusion[i][j] != curr_confusion[i][j]:
                        changed = True
                        break

        return changed, curr_confusion

    def _calc_confusion_matrix(self, y_true, pred):
        results = [[0 for i in range(self.output_units)] for j in range(self.output_units)]
        for idx, p in enumerate(pred):
            label = self._get_label(y_true[idx])
            predict = int(p)
            results[label][predict] += 1
        return results

    def _feedback(self, cluster, feature, label):
        self._duplicate_delete(cluster, feature)

        # update
        self.clusters["clusters"][cluster]["features"].append(feature)
        self.clusters["clusters"][cluster]["labels"].append(label)

        self.clusters["clusters"][cluster]["stats"] = self._compute_cluster_stats(
            self.clusters["clusters"][cluster]["stats"], label
        )
        self.clusters["clusters"][cluster]["knn"] = BallTree(
            np.array(self.clusters["clusters"][cluster]["features"])
        )

    def _duplicate_delete(self, cluster, feature):
        duplicate = True
        while duplicate:
            duplicate = False
            for idx, dst_feature in enumerate(
                self.clusters["clusters"][cluster]["features"]):
                if self._calculate_distance(dst_feature, feature) == 0.0:
                    self.clusters["clusters"][cluster]["features"].pop(idx)
                    label = self.clusters["clusters"][cluster]["labels"][idx]
                    self.clusters["clusters"][cluster]["labels"].pop(idx)
                    self.clusters["clusters"][cluster]["stats"][label] -= 1
                    duplicate = True
                    break

    def _knn_classifier(self, feature, cluster: Dict):
        labels_cluster = cluster.get("labels")
        cluster_stat = list(cluster.get("stats").keys())

        pred = [0.0 for i in range(self.output_units)]

        if len(cluster_stat) == 1:
            pred[int(cluster_stat[0])] = 1.0
        else:
            d, p = cluster.get("knn").query([feature], k=1, return_distance=True)
            dist = d[0][0]
            if dist != 0:
                max_dist = cluster.get("max_dist", 0.0)
                if dist > max_dist:
                    cluster["max_dist"] = dist
                    max_dist = dist
                dist /= (max_dist * 1.1)

            pred[labels_cluster[p[0][0]]] = 1.0 - dist
        return pred

    def saved_model(self):
        dir_model = '{}/{}/{}'.format(
            Constants.DIR_TEMP, self.param_dict["model_nm"], self.param_dict["alg_sn"]
        )

        if not FileUtils.is_exist(dir_model):
            FileUtils.mkdir(dir_model)

        # back-born save
        self.backborn.model.save(dir_model + "/backborn.h5")
        self.backborn.model.save_weights(dir_model + "/backborn-weight.h5")

        # nearest feature save
        f = open("{}/clusters.model".format(dir_model), "wb")
        try:
            pickle.dump(self.clusters, f)
        except Exception as e:
            self.LOGGER.error(e, exc_info=True)
        finally:
            f.close()

        self.LOGGER.info(f"{self.ALG_CODE} Saved...")
        self.LOGGER.info("model dir : {}".format(dir_model))

    def load_model(self):
        dir_model = '{}/{}/{}'.format(
            Constants.DIR_TEMP, self.param_dict["model_nm"], self.param_dict["alg_sn"]
        )

        if not os.path.exists(dir_model):
            dir_model = '{}/{}/{}'.format(
                Constants.DIR_LOAD_MODEL, self.param_dict["model_nm"], self.param_dict["alg_sn"]
            )

        backborn_path = dir_model
        if FileUtils.is_exist(backborn_path):
            # back-born load
            # self.backborn.model = tf.keras.models.load_model(dir_model + "/backborn.h5")
            self.backborn.model.load_weights(backborn_path + "/backborn-weight.h5")
            self.LOGGER.info(f"backborn model dir : {backborn_path}/backborn-weight.h5")

        cluster_path = "{}/clusters.model".format(dir_model)
        if FileUtils.is_exist(cluster_path):
            # nearest feature load
            f = open(cluster_path, "rb")
            try:
                self.clusters = pickle.load(f)
            except Exception as e:
                self.LOGGER.error(e, exc_info=True)
            finally:
                f.close()
            self.LOGGER.info(f"{self.ALG_CODE} Loaded...")
            self.LOGGER.info(f"cluster model dir : {cluster_path}")

    # member functions
    def _make_cluster(self, ball_tree_pred, extracted_features, labels) -> Dict:
        result = {"centroid": list(), "clusters": list()}

        index_dict = dict()
        for idx, idx_list in enumerate(ball_tree_pred):
            # Clusters
            cluster_dict = {"features": list(), "labels": list(),
                            "stats": dict(), "knn": None}
            for i in idx_list:
                if index_dict.get(i, None) is None:
                    index_dict[i] = True
                    feature = extracted_features[i]
                    label = self._get_label(labels[i])

                    if not self._check_duplicate(cluster_dict["features"], feature):
                        cluster_dict["features"].append(feature)
                        cluster_dict["labels"].append(label)

                        cluster_dict["stats"] = self._compute_cluster_stats(cluster_dict.get("stats"), label)

            # Means
            if len(cluster_dict.get("features", list())) > 0:
                centroid = self._compute_centroid(cluster_dict.get("features"))
                if not self._check_duplicate(result.get("centroid"), centroid):
                    cluster_dict["knn"] = BallTree(np.array(cluster_dict.get("features")))
                    result["centroid"].append(centroid)
                    result["clusters"].append(cluster_dict)

            if idx % 1000 == 0:
                self.LOGGER.info("make cluster process ... : {}".format(idx))

        return result

    @staticmethod
    def _get_label(label):
        if len(label) != 1:
            return int(np.argmax(label))
        else:
            return int(label[0])

    @staticmethod
    def _compute_cluster_stats(cluster_dict, label):
        # # label statistics
        if cluster_dict.get(label, None) is None:
            cluster_dict[label] = 1
        else:
            cluster_dict[label] += 1
        return cluster_dict

    @staticmethod
    def _compute_centroid(features):
        return np.mean(features, axis=0).tolist()

    @classmethod
    def _check_duplicate(cls, dst_feature_list, src_feature) -> bool:
        for dst_feature in dst_feature_list:
            if cls._calculate_distance(dst_feature, src_feature) == 0.0:
                return True
        return False

    @staticmethod
    def _calculate_distance(dst_feature, src_feature):
        return np.linalg.norm(np.array(dst_feature) - np.array(src_feature))

    def print_cluster_statistics(self, clusters):
        total = 0
        unique = 0
        for idx, c in enumerate(clusters):
            if len(c.get("stats").keys()) == 1:
                unique += 1
            self.LOGGER.debug("cluster # : {}, labels: {}".format(
                idx, c.get("stats")
            ))
            total += 1

        self.LOGGER.info("total : {}, unique : {}, ratio: {}".format(
            total, unique, float(unique / total)
        ))

    def eval_regressor(self, dataset):
        raise NotImplementedError

    def eval_we(self, dataset):
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    def eval_clustering(self, dataset):
        raise NotImplementedError

    def learn_result_regressor(self, data):
        raise NotImplementedError
