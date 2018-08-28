
import os
import pandas as pd

class FileUtility:

    @staticmethod
    def read_excel(filename, sheet, **args):
        if not os.path.isfile(filename):
            raise Exception("File {0} does not exist!".format(filename))
        with pd.ExcelFile(filename) as xls:
            return pd.read_excel(xls, sheet, **args)

    @staticmethod
    def save_excel(data, filename):
        with pd.ExcelWriter(filename) as writer:
            for (df, name) in data:
                df.to_excel(writer, name, engine='xlsxwriter')
            writer.save()
            