#  -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.

import os

from pycmmn.Singleton import Singleton
from pycmmn.tools.VersionManagement import VersionManagement
from pycmmn.utils.ConfUtils import ConfUtils
from pycmmn.utils.FileUtils import FileUtils


class Constants(object, metaclass=Singleton):
    _working_dir = os.getcwd()
    _data_cvt_dir = _working_dir + "/../apeflow"
    _conf_xml_filename = _data_cvt_dir + "/conf/apeflow-conf.xml"

    _MODE = "deploy"

    if not FileUtils.is_exist(_conf_xml_filename):
        _MODE = "dev"

        if _working_dir != "/eyeCloudAI/app/ape/apeflow":
            os.chdir(FileUtils.get_realpath(__file__) + "/../../")

        _working_dir = os.getcwd()
        _data_cvt_dir = _working_dir + "/../apeflow"
        _conf_xml_filename = _working_dir + "/conf/apeflow-conf.xml"

    _CONFIG = ConfUtils.load(filename=_conf_xml_filename)

    try:
        VERSION_MANAGER = VersionManagement(app_path=_working_dir)
    except Exception as e:
        # DEFAULT
        VersionManagement.generate(
            version="1.0.0",
            app_path=_working_dir,
            module_nm="apeflow",
        )
        VERSION_MANAGER = VersionManagement(app_path=_working_dir)
    VERSION = VERSION_MANAGER.VERSION
    MODULE_NM = VERSION_MANAGER.MODULE_NM

    DIR_APP = _CONFIG.get("app_dir", "/eyeCloudAI")
    DIR_DATA_ROOT = DIR_APP + _CONFIG.get("dir_data_root", "/data")
    DIR_PROCESSING = DIR_DATA_ROOT + _CONFIG.get("dir_processing", "/processing/ape")
    DIR_STORAGE = DIR_DATA_ROOT + _CONFIG.get("dir_storage", "/storage/ape")
    DIR_TEMP = DIR_PROCESSING + _CONFIG.get("dir_temp", "/temp")
    DIR_RESOURCES = FileUtils.get_realpath(file=__file__) + "/../resources"

    # LOG SETTING
    DIR_LOG = DIR_APP + _CONFIG.get("log_dir", "/logs")
    LOG_NAME = _CONFIG.get("log_name", "ApeFlow")
    LOG_LEVEL = _CONFIG.get(
        "log_level", "INFO"
    )  # one of [INFO, DEBUG, WARN, ERROR, CRITICAL]

    REST_URL_ROOT = "http://{}:{}".format(
        _CONFIG.get("mrms_svc", "mrms-svc"),
        _CONFIG.get("mrms_rest_port", "9200"),
    )

    # Hosts
    MRMS_SVC = _CONFIG.get("mrms_svc", "mrms-svc")
    MRMS_USER = _CONFIG.get("mrms_username", "HE12RmzKHQtH3bL7tTRqCg==")
    MRMS_PASSWD = _CONFIG.get("mrms_password", "jTf6XrqcYX1SAhv9JUPq+w==")
    MRMS_SFTP_PORT = int(_CONFIG.get("mrms_sftp_port", "10022"))
    MRMS_REST_PORT = int(_CONFIG.get("mrms_rest_port", "9200"))

    EARLY_TYPE_NONE = "0"
    EARLY_TYPE_MIN = "1"
    EARLY_TYPE_MAX = "2"
    EARLY_TYPE_VAR = "3"

    TF = "TF"
    KERAS = "Keras"
    TFV1 = "TFV1"
    TF_BACKEND_LIST = [TF, KERAS, TFV1]
    TF_BACKEND_V1 = "TFv1"
    TF_BACKEND_V2 = "TFv2"
    TF_BACKEND_NONE = "None"
    GENSIM = "GS"
    SCIKIT_LEARN = "SKL"
    APEFLOW = "APE"
    PYTORCH = "PyTorch"

    LIB_TYPE_TF = "1"
    LIB_TYPE_GS = "4"
    LIB_TYPE_SKL = "5"

    DIST_TYPE_SINGLE = "single"
    DIST_TYPE_DISTRIBUTE = "distribute"

    OUT_MODEL_PB = "pb"
    OUT_MODEL_TF = "tf"
    OUT_MODEL_JSON = "json"
    OUT_MODEL_PKL = "pkl"
    OUT_MODEL_JAVA = "java"
    OUT_MODEL_FOLDER = "folder"
    OUT_MODEL_HYBRID = "hybrid"
    OUT_MODEL_HYBRID_TF = "hybridTF"
    OUT_MODEL_KERAS_TOKENIZER = "kToken"
    OUT_MODEL_APE_OUTLIER_DETCTION = "APE_OUTLIER_DETECTION"
    OUT_MODEL_ONNX = "onnx"  # https://onnx.ai/
    OUT_MODEL_PYTORCH = "PyTorch"

    BATCH_SIZE = 1024


if __name__ == "__main__":
    print(Constants.__dict__)
