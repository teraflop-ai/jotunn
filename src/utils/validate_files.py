import os

import pyarrow.parquet as pq
from tqdm import tqdm


class ValidateParquets:
    def __init__(self, extension: str = ".parquet"):
        self.extension = extension

    def scan_dir(self, directory):
        valid_parquets = []
        invalid_parquets = []
        for file in tqdm(
            self.get_filepaths(directory), desc="Validating Parquet Files"
        ):
            try:
                pq.ParquetFile(file).schema
                valid_parquets.append(file)
            except Exception as e:
                invalid_parquets.append(file)
        return valid_parquets, invalid_parquets

    def get_filepaths(self, directory):
        file_paths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if self.extension in filename:
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)
        return file_paths

    def remove_invalid_files(self, file_paths):
        for file in file_paths:
            try:
                os.remove(file)
            except Exception as e:
                print(e)


def prepare_parquets(directory):
    validator = ValidateParquets(".parquet")
    valid_files, invalid_files = validator.scan_dir(directory)
    validator.remove_invalid_files(invalid_files)
    return valid_files
