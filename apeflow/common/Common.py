#  -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.
#

import json

from pycmmn.Singleton import Singleton
from pycmmn.logger.MPLogger import MPLogger
from pycmmn.utils.FileUtils import FileUtils
from apeflow.common.Constants import Constants


class Common(object, metaclass=Singleton):
    # MAKE DIR
    FileUtils.mkdir(Constants.DIR_LOG)
    FileUtils.mkdir(Constants.DIR_DATA_ROOT)
    FileUtils.mkdir(Constants.DIR_PROCESSING)
    FileUtils.mkdir(Constants.DIR_STORAGE)
    FileUtils.mkdir(Constants.DIR_TEMP)

    # LOG SETTING
    LOGGER = MPLogger(log_name=Constants.LOG_NAME, log_level=Constants.LOG_LEVEL, log_dir=Constants.DIR_LOG)
    LOGGER.getLogger().info("APEFlow v.%s APEFlow Logger initialized..." % Constants.VERSION)

    with open(Constants.DIR_RESOURCES + "/com_code.json", "r") as f:
        COM_CODE = json.load(f)

    with open(Constants.DIR_RESOURCES + "/com_func.json", "r") as f:
        COMMON_FUNC = json.load(f)

    ACTIVATE_FN_CODE_DICT = COMMON_FUNC.get("activation_fn_code")
    CONV_FN_CODE_DICT = COMMON_FUNC.get("conv_fn_code")
    POOLING_FN_CODE_DICT = COMMON_FUNC.get("pooling_fn_code")
    UPSAMPLING_FN_CODE_DICT = COMMON_FUNC.get("upsample_fn_code")
    OPTIMIZER_FN_CODE_DICT = COMMON_FUNC.get("optimizer_fn_code")


if __name__ == '__main__':
    Common()
