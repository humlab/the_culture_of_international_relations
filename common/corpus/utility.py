import fnmatch
import os
import re
import zipfile
from collections.abc import Callable, Generator
from typing import Any, AnyStr, Self

from loguru import logger
import pandas as pd

HYPHEN_REGEXP: re.Pattern = re.compile(r"\b(\w+)-\s*\r?\n\s*(\w+)\b", re.UNICODE)


def any_to_unicode(raw_text: AnyStr, encoding: str = "utf8", errors: str = "strict") -> str:
    return raw_text if isinstance(raw_text, str) else str(raw_text, encoding, errors=errors)


def dehyphen(text: str) -> str:
    result: str = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result


def list_archive_files(archivename: str, pattern: str | re.Pattern | Callable[[str], bool]) -> list[str]:
    def px(x: str) -> bool:
        if isinstance(pattern, str):
            return fnmatch.fnmatch(x, pattern)
        if isinstance(pattern, re.Pattern):
            return bool(pattern.match(x))
        if callable(pattern):
            return bool(pattern(x))
        return False

    with zipfile.ZipFile(archivename) as zf:
        return [name for name in zf.namelist() if px(name)]


class CompressedFileReader:

    def __init__(
        self,
        path: str,
        pattern: str | re.Pattern = "*.txt",
        itemfilter: re.Pattern | str | Callable[[str], bool] | None = None,
    ) -> None:
        self.path: str = path
        self.filename_pattern: str | re.Pattern = pattern
        self.archive_filenames: list[str] = list_archive_files(path, pattern)
        filenames: list[str] | None = None
        if itemfilter is not None:
            if isinstance(itemfilter, list):
                filenames = [x for x in itemfilter if x in self.archive_filenames]
            elif callable(itemfilter):
                filenames = [x for x in self.archive_filenames if itemfilter(x)]
            else:
                raise ValueError("itemfilter must be a list or callable")
        self.filenames: list[str] = filenames or self.archive_filenames
        self.iterator: Generator[tuple[str, str], Any, None] | None = None

    def __iter__(self) -> Self:
        self.iterator = None
        return self

    def __next__(self) -> tuple[Any | str, str]:
        if self.iterator is None:
            self.iterator = self.get_iterator()
        return next(self.iterator)

    def get_file(self, filename: str) -> Generator[tuple[str, None] | tuple[str, str], Any, None]:

        if filename not in self.filenames:
            yield os.path.basename(filename), None

        with zipfile.ZipFile(self.path) as zip_file:
            yield os.path.basename(filename), self._read_content(zip_file, filename)

    def get_iterator(self) -> Generator[tuple[str, str], Any, None]:
        with zipfile.ZipFile(self.path) as zip_file:
            for filename in self.filenames:
                yield os.path.basename(filename), self._read_content(zip_file, filename)

    def _read_content(self, zip_file: zipfile.ZipFile, filename: str) -> str:
        with zip_file.open(filename, "r") as text_file:
            byte_str: bytes = text_file.read()
            content: str = any_to_unicode(byte_str, encoding="utf8", errors="ignore")
            content = dehyphen(content)
            return content


def get_document_stream(source: CompressedFileReader | str, lang: str, document_index: pd.DataFrame | None = None):

    assert document_index is not None

    if "document_id" not in document_index.columns:
        document_index["document_id"] = document_index.index

    def id_extractor(filename: str) -> str:
        match: re.Match | None = re.match(r"^(\w*)\_" + lang + r"([\_\-]corr)?\.txt$", filename)
        if match:
            return match.group(1)
        return ""

    lang_pattern: re.Pattern = re.compile(rf"^(\w*)_{lang}([_-]corr)?\.txt$")

    def item_filter(x):
        return lang_pattern.match(x)  # and id_extractor(x) in document_index.index

    if isinstance(source, str):
        print(f"Opening archive: {source}")
        reader: CompressedFileReader = CompressedFileReader(source, pattern=lang_pattern, itemfilter=item_filter)
    else:
        reader = source

    id_map: dict[str, str] = {
        filename: id_extractor(filename) for filename in reader.filenames if item_filter(filename)
    }

    if len(set(document_index.index) - set(id_map.values())) > 0:
        logger.warning(
            "Treaties not found in archive: " + ", ".join(list(set(document_index.index) - set(id_map.values())))
        )

    columns: list[str] = ["signed_year", "party1", "party2"]

    df: pd.DataFrame = document_index[columns]

    for filename, text in reader:

        document_id: str | None = id_map.get(filename)

        if document_id not in df.index:
            continue

        metadata = df.loc[document_id].to_dict()

        metadata["filename"] = filename
        metadata["document_id"] = document_id
        metadata["treaty_id"] = document_id

        yield filename, document_id, text, metadata
