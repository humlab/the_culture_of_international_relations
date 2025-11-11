import fnmatch
import os
import re
import zipfile
from collections.abc import Callable, Generator
from typing import Any, Self

import gensim

HYPHEN_REGEXP: re.Pattern = re.compile(r"\b(\w+)-\s*\r?\n\s*(\w+)\b", re.UNICODE)


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
        self, path: str, pattern: str = "*.txt", itemfilter: re.Pattern | str | Callable[[str], bool] | None = None
    ) -> None:
        self.path: str = path
        self.filename_pattern: str = pattern
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
            content: str = gensim.utils.to_unicode(byte_str, "utf8", errors="ignore")
            content = dehyphen(content)
            return content
