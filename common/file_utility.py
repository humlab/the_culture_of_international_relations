import os
import shutil
import time
import zipfile
from ctypes import ArgumentError

import pandas as pd


class FileUtility:

    @staticmethod
    def create(directory: str, clear_target_dir: bool = False) -> None:

        if os.path.exists(directory) and clear_target_dir:
            shutil.rmtree(directory)

        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def read_excel(filename: str, sheet: str) -> pd.DataFrame:
        if not os.path.isfile(filename):
            raise ArgumentError(f"File {filename} does not exist!")
        with pd.ExcelFile(filename) as xls:
            return pd.read_excel(xls, sheet)

    @staticmethod
    def save_excel(data: list[tuple[pd.DataFrame, str]], filename: str) -> None:
        with pd.ExcelWriter(filename) as writer:
            for df, name in data:
                df.to_excel(writer, name, engine="xlsxwriter")
            writer.save()

    @staticmethod
    def data_path(directory, filename):
        return os.path.join(directory, filename)

    @staticmethod
    def ts_data_path(directory: str, filename: str) -> str:
        return os.path.join(
            directory, f"{time.strftime('%Y%m%d%H%M')}_{filename}"
        )

    @staticmethod
    def data_path_ts(directory: str, path: str) -> str:
        basename, extension = os.path.splitext(path)
        return os.path.join(
            directory,
            f"{basename}_{time.strftime('%Y%m%d%H%M')}{extension}",
        )

    @staticmethod
    def zip(path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist!")
        folder, filename = os.path.split(path)
        basename, _ = os.path.splitext(filename)
        zip_name: str = os.path.join(folder, basename + ".zip")
        with zipfile.ZipFile(
            zip_name, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(path)
        os.remove(path)
