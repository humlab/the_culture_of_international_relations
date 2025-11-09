import logging
import os

import pandas as pd
import requests

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


def download_file_from_google_drive(gid: str, destination: str) -> None:
    def get_confirm_token(response: requests.Response) -> str | None:
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        logger.info(f"Stored: {destination}")
        print(f"Stored: {destination}")

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response: requests.Response = session.get(URL, params={"id": gid}, stream=True)
    token: str | None = get_confirm_token(response)

    if token:
        params: dict[str, str] = {"id": gid, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def extract_sheets(path: str, sheets: list[str]) -> None:  # pylint: disable=unused-argument

    folder, filename = os.path.split(path)
    basename, _ = os.path.splitext(filename)

    with pd.ExcelFile(path) as xls:
        data: dict[str, pd.DataFrame] = pd.read_excel(xls, sheet_name=None)

    for sheet_name in data.keys():

        if not sheet_name in data.keys():
            continue

        df: pd.DataFrame = data[sheet_name]

        if not hasattr(df, "to_csv"):
            continue

        csv_name: str = os.path.join(folder, f"{basename}_{sheet_name}.csv")

        if os.path.exists(csv_name):
            os.remove(csv_name)

        df.to_csv(csv_name, sep="\t")

        logger.info(f"Extracted: {csv_name}")
        print(f"Extracted: {csv_name}")


def process_file(file: dict[str, str], overwrite: bool = False) -> None:

    print(f"Processing: {file['file_id']}")
    if overwrite and os.path.exists(file["destination"]):
        os.remove(file["destination"])
        logger.info(f"Removed: {file['destination']}")
    else:
        print("Skipping. File exists in ./data!")

    # if not os.path.exists(file['destination']):
    print(f"Downloading: {file['file_id']}")
    download_file_from_google_drive(file["file_id"], file["destination"])

    if len(file["sheets"] or []) > 0:
        extract_sheets(file["destination"], file["sheets"])


def process_files(files_to_download: list[dict[str, str]]) -> None:

    for file in files_to_download:
        process_file(file)
