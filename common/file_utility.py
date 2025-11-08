from ctypes import ArgumentError
import os
import time
import shutil
import zipfile
import pandas as pd


class FileUtility:

    @staticmethod
    def create(directory, clear_target_dir=False):

        if os.path.exists(directory) and clear_target_dir:
            shutil.rmtree(directory)

        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def read_excel(filename, sheet):
        if not os.path.isfile(filename):
            raise ArgumentError(f"File {filename} does not exist!")
        with pd.ExcelFile(filename) as xls:
            return pd.read_excel(xls, sheet)

    @staticmethod
    def save_excel(data, filename):
        with pd.ExcelWriter(filename) as writer:
            for df, name in data:
                df.to_excel(writer, name, engine="xlsxwriter")
            writer.save()

    @staticmethod
    def data_path(directory, filename):
        return os.path.join(directory, filename)

    @staticmethod
    def ts_data_path(directory, filename):
        return os.path.join(
            directory, "{}_{}".format(time.strftime("%Y%m%d%H%M"), filename)
        )

    @staticmethod
    def data_path_ts(directory, path):
        basename, extension = os.path.splitext(path)
        return os.path.join(
            directory,
            "{}_{}{}".format(basename, time.strftime("%Y%m%d%H%M"), extension),
        )

    @staticmethod
    def zip(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist!")
        folder, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        zip_name = os.path.join(folder, basename + ".zip")
        with zipfile.ZipFile(
            zip_name, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(path)
        os.remove(path)

        from numpy import random as rnd
