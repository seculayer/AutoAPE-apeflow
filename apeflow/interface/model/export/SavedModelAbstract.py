# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.co.kr
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

import os
from typing import Callable
import json

from apeflow.common.Constants import Constants
from apeflow.common.Common import Common
from pycmmn.utils.FileUtils import FileUtils
from pycmmn.exceptions.NotSupportTypeError import NotSupportTypeError
from pycmmn.sftp.SFTPClientManager import SFTPClientManager


class SavedModelAbstract(object):
    LOGGER = Common.LOGGER.getLogger()
    try:
        MRMS_SFTP_MANAGER: SFTPClientManager = SFTPClientManager(
            "{}:{}".format(Constants.MRMS_SVC, Constants.MRMS_SFTP_PORT),
            Constants.MRMS_USER, Constants.MRMS_PASSWD, LOGGER
        )
    except Exception as e:
        LOGGER.error(e, exc_info=True)
        LOGGER.info(f"[Apeflow] svc : {Constants.MRMS_SVC}, sftp port: {Constants.MRMS_SFTP_PORT}")
        LOGGER.info(f"[Apeflow] user name : {Constants.MRMS_USER}, passwd: {Constants.MRMS_PASSWD}")

    @classmethod
    def init(cls, dir_model, param_dict):
        if cls._check_dir_model(dir_model):
            try:
                # backup
                FileUtils.move_dir(dir_model, dir_model + "_prev")
                FileUtils.mkdir('{}/{}/{}'.format(Constants.DIR_TEMP, param_dict["model_nm"], param_dict["alg_sn"]))
            except Exception as e:
                Common.LOGGER.getLogger().error(str(e), exc_info=True)
                raise e
        else:
            if not FileUtils.is_exist('{}/{}'.format(Constants.DIR_TEMP, param_dict["model_nm"])):
                FileUtils.mkdir('{}/{}'.format(Constants.DIR_TEMP, param_dict["model_nm"]))
                FileUtils.mkdir('{}/{}/{}'.format(Constants.DIR_TEMP, param_dict["model_nm"], param_dict["alg_sn"]))

    @classmethod
    def save(cls, model):
        param_dict = model.param_dict

        dir_model = '{}/{}/{}'.format(
            Constants.DIR_TEMP, param_dict["model_nm"], param_dict["alg_sn"]
        )
        try:
            cls.init(dir_model, param_dict)
            case: Callable = cls._save_case_fn(model)
            cls._save_except_execute(dir_model, model)

            try:
                case(model, dir_model)
            except Exception as e:
                cls.LOGGER.error(e, exc_info=True)
                raise NotSupportTypeError(algorithm_type=model.OUT_MODEL_TYPE)

            if model.task_idx == 0:
                cls._scp_model_to_storage(dir_model, param_dict)
            cls._remove_backup(dir_model + "_prev")
            cls._remove_backup(dir_model)

            cls.LOGGER.info("model saved ....")
            cls.LOGGER.info("model dir : {}".format(dir_model))
        except Exception as e:
            cls.LOGGER.error(str(e), exc_info=True)
            # RESTORE
            try:
                FileUtils.remove_dir(dir_model)
                FileUtils.move_dir(dir_model + "_prev", dir_model)
            except Exception as e:
                cls.LOGGER.error(str(e), exc_info=True)
                raise e

    @classmethod
    def _save_except_execute(cls, dir_model, model):
        if model.ALG_CODE == "KDAEOD":
            result_dict = {
                "rmse": model.rmse
            }
            f = FileUtils.file_pointer("{}/apeflow_json.tmp".format(dir_model), "w")
            try:
                f.write(
                    json.dumps(result_dict, indent=4)
                )
                f.write("\n")
            except Exception as e:
                cls.LOGGER.error(e, exc_info=True)
            finally:
                f.close()

            os.rename("{}/apeflow_json.tmp".format(dir_model), "{}/apeflow_json.model".format(dir_model))

    @classmethod
    def _scp_model_to_storage(cls, dir_model: str, param_dict: dict) -> None:
        remote_path = f"{Constants.DIR_STORAGE}/{param_dict['model_nm']}"
        if not cls.MRMS_SFTP_MANAGER.is_exist(remote_path):
            cls.MRMS_SFTP_MANAGER.mkdir(remote_path)
        cls.MRMS_SFTP_MANAGER.scp_to_storage(
            dir_model, remote_path
        )

    @classmethod
    def _scp_model_from_storage(cls, dir_model: str, param_dict: dict) -> None:
        remote_path = f"{Constants.DIR_STORAGE}/{param_dict['model_nm']}"
        try:
            cls.MRMS_SFTP_MANAGER.scp_from_storage(
                remote_path, dir_model
            )
        except Exception as e:
            pass

    @classmethod
    def _save_case_fn(cls, model) -> Callable:
        raise NotImplementedError

    @classmethod
    def _remove_backup(cls, dir_model) -> None:
        # REMOVE BACKUP-MODEL
        try:
            if cls._check_dir_model(dir_model):
                FileUtils.remove_dir(dir_model)
        except Exception as e:
            cls.LOGGER.error(str(e), exc_info=True)
            raise e

    @classmethod
    def _check_dir_model(cls, dir_model):
        return os.path.exists(dir_model)

    @classmethod
    def load(cls, model):
        param_dict = model.param_dict
        cls._scp_model_from_storage(Constants.DIR_TEMP, param_dict)
        dir_model = '{}/{}/{}'.format(
            Constants.DIR_TEMP, param_dict["model_nm"], param_dict["alg_sn"]
        )
        cls._load_except_execute(dir_model, model)

        if cls._check_dir_model(dir_model):
            try:
                case: Callable = cls._load_case_fn(model)
                try:
                    case(model, dir_model)
                except Exception as e:
                    cls.LOGGER.error(e, exc_info=True)
                    raise NotSupportTypeError(algorithm_type=model.OUT_MODEL_TYPE)

                cls.LOGGER.info("model load ....")
                cls.LOGGER.info("model dir : {}".format(dir_model))
            except Exception as e:
                cls.LOGGER.warn(str(e), exc_info=True)
                raise e
        else:
            cls.LOGGER.warn("MODEL FILE IS NOT EXIST : [{}]".format(dir_model))

    @classmethod
    def _load_except_execute(cls, dir_model, model):
        pass
        # if model.ALG_CODE == "KDAEOD":
        #     try:
        #         f = FileUtils.file_pointer("{}/apeflow_json.model".format(dir_model), "r")
        #         try:
        #             result_dict = json.load(f)
        #             model.rmse = result_dict.get("rmse", 0)
        #
        #         except Exception as e:
        #             cls.LOGGER.error(e, exc_info=True)
        #         finally:
        #             f.close()
        #     except Exception as e:
        #         cls.LOGGER.warn("apeflow_json.model file is not existed ...")

    @classmethod
    def _load_case_fn(cls, model) -> Callable:
        raise NotImplementedError
