# -*- coding: utf-8 -*-
import functools
import glob
import inspect
import logging
import os
import platform
import re
import string
import sys
import time
import types
import zipfile
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import wordcloud

# pylint: disable=redefined-builtin


def getLogger(name: str = "cultural_treaties", level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(filename)s.%(funcName)s() : %(message)s",
        level=level,
    )
    _logger: logging.Logger = logging.getLogger(name=name)
    _logger.setLevel(level=level)
    return _logger


logger: logging.Logger = getLogger(name=__name__)

__cwd__: str = os.path.abspath(path=__file__) if "__file__" in globals() else os.getcwd()

sys.path.append(__cwd__)


def remove_snake_case(snake_str: str) -> str:
    return " ".join(x.title() for x in snake_str.split("_"))


def noop(*args) -> None:  # pylint: disable=W0613
    pass


def isint(s: str) -> bool:
    try:
        int(s)
        return True
    except:  # pylint: disable=bare-except
        return False


def filter_dict(d: dict[Any, Any], keys: set[Any] | None = None, filter_out: bool = False) -> dict[Any, Any]:
    keys = set(d.keys()) - set(keys or []) if filter_out else (keys or [])  # type: ignore
    return {k: v for k, v in d.items() if k in keys}


def timecall(f):

    @functools.wraps(f)
    def f_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = f(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Call time [{f.__name__}]: {elapsed:.4f} secs")
        return value

    return f_wrapper


def extend(target: dict, *args: dict, **kwargs: dict) -> dict:
    """Returns dictionary 'target' extended by supplied dictionaries (args) or named keywords

    Parameters
    ----------
    target : dict
        Default dictionary (to be extended)

    args: [dict]
        Optional. List of dicts to use when updating target

    args: [key=value]
        Optional. List of key-value pairs to use when updating target

    Returns
    -------
    [dict]
        Target dict updated with supplied dicts/key-values.
        Multiple keys are overwritten inorder of occrence i.e. keys to right have higher precedence

    """

    target = dict(target)
    for source in args:
        target.update(source)
    target.update(kwargs)
    return target


def ifextend(target: dict, source: dict, p: bool) -> dict:
    return extend(target=target, source=source) if p else target


def extend_single(target: dict, source: dict, name: str) -> dict:
    if name in source:
        target[name] = source[name]
    return target


class SimpleStruct(types.SimpleNamespace):
    """A simple value container based on built-in SimpleNamespace."""

    def update(self, **kwargs) -> types.NoneType:
        self.__dict__.update(kwargs)


def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    """Returns a flat single list out of supplied list of lists."""

    return [item for sublist in list_of_lists for item in sublist]


def project_series_to_range(series: pd.Series, low: float, high: float) -> pd.Series:
    """Project a sequence of elements to a range defined by (low, high)"""
    norm_series: pd.Series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)


def project_to_range(value: float, low: float, high: float) -> float:
    """Project a singlevalue to a range (low, high)"""
    return low + (high - low) * value


def clamp_values(values: list[float], low_high: tuple[float, float]) -> list[float]:
    """Clamps value to supplied interval."""
    mw: float = max(values)
    return [project_to_range(w / mw, low_high[0], low_high[1]) for w in values]


def filter_kwargs(f, args) -> dict[Any, Any] | Any:
    """Removes keys in dict arg that are invalid arguments to function f

    Parameters
    ----------
    f : [fn]
        Function to introspect
    args : dict
        List of parameter names to test validity of.

    Returns
    -------
    dict
        Dict with invalid args filtered out.
    """

    try:
        return {k: args[k] for k in args.keys() if k in inspect.signature(f).parameters}
    except:  # pylint: disable=W0702
        return args


VALID_CHARS: str = "-_.() " + string.ascii_letters + string.digits


def filename_whitelist(filename: str) -> str:
    filename = "".join(x for x in filename if x in VALID_CHARS)
    return filename


def cpif_deprecated(source: dict, target: dict, name: str) -> dict:
    logger.debug(msg="use of cpif is deprecated")
    if name in source:
        target[name] = source[name]
    return target


def dict_subset(d: dict, keys: set) -> dict:
    if keys is None:
        return d
    return {k: v for (k, v) in d.items() if k in keys}


# def dict_split(d: dict, fn: Callable[[dict], bool]) -> tuple[dict[Any, Any], dict[Any, Any]]:
#     """Splits a dictionary into two parts based on predicate"""
#     true_keys = {k for k in d.keys() if fn(d, k)}
#     return {k: d[k] for k in true_keys}, {k: d[k] for k in set(d.keys()) - true_keys}


def list_of_dicts_to_dict_of_lists(
    list_of_dicts: list[dict],
) -> dict[Any, tuple[Any, ...]]:
    dict_of_lists = dict(zip(list_of_dicts[0], zip(*[d.values() for d in list_of_dicts])))
    return dict_of_lists


def uniquify(sequence: list[Any]) -> list[Any]:
    """Removes duplicates from a list whilst still preserving order"""
    seen: set[Any] = set()
    seen_add: Callable[[Any], None] = seen.add
    return [x for x in sequence if not (x in seen or seen_add(x))]


def sort_chained(x: list[Any], f: Callable[[Any], Any]) -> list[Any]:
    return sorted(x, key=f, reverse=True)


def ls_sorted(path: str) -> list[str]:
    return sort_chained(list(filter(os.path.isfile, glob.glob(pathname=path))), f=os.path.getmtime)


# def split(delimiters: list[str], string: str, maxsplit: int = 0) -> list[str]:
#     regexPattern = "|".join(map(re.escape, delimiters))
#     return re.split(regexPattern, string, maxsplit)


HYPHEN_REGEXP: re.Pattern = re.compile(r"\b(\w+)-\s*\r?\n\s*(\w+)\b", flags=re.UNICODE)


def dehyphen(text: str) -> str:
    result: str = re.sub(pattern=HYPHEN_REGEXP, repl=r"\1\2\n", string=text)
    return result


# path = types.SimpleNamespace()


def path_add_suffix(path: str, suffix: str, new_extension: str | None = None) -> str:
    basename, extension = os.path.splitext(path)
    suffixed_path: str = basename + suffix + (extension if new_extension is None else new_extension)
    return suffixed_path


def path_add_timestamp(path: str, fmt: str = "%Y%m%d%H%M") -> str:
    suffix: str = f"_{time.strftime(fmt)}"
    return path_add_suffix(path, suffix)


def path_add_date(path: str, fmt: str = "%Y%m%d") -> str:
    suffix: str = f"_{time.strftime(fmt)}"
    return path_add_suffix(path, suffix)


def path_add_sequence(path: str, i: int, j: int = 0) -> str:
    suffix: str = str(i).zfill(j)
    return path_add_suffix(path, suffix)


def zip_get_filenames(zip_filename: str, extension: str = ".txt") -> list[str]:
    with zipfile.ZipFile(zip_filename, mode="r") as zf:
        return [x for x in zf.namelist() if x.endswith(extension)]


def zip_get_text(zip_filename: str, filename: str) -> str:
    with zipfile.ZipFile(zip_filename, mode="r") as zf:
        return zf.read(filename).decode(encoding="utf-8")


def slim_title(x: str) -> str:
    try:
        m: re.Match[str] | types.NoneType = re.match(r".*\((.*)\)$", x)  # pylint: disable=W1401
        if m is not None:
            g: tuple[str | Any, ...] = m.groups()
            return g[0]
        return " ".join(x.split(" ")[:3]) + "..."
    except:  # pylint: disable=W0702
        return x


def complete_value_range(values: list[Any], typef: type) -> list[Any] | list[str]:
    """Create a complete range from min/max range in case values are missing

    Parameters
    ----------
    str_values : list
        List of values to fill

    Returns
    -------
    """

    if len(values) == 0:
        return []

    int_values: list[int] = list(map(int, values))
    int_values = list(range(min(int_values), max(int_values) + 1))

    return list(map(typef, int_values))


def is_platform_architecture(xxbit):

    assert xxbit in ["32bit", "64bit"]
    logger.info(platform.architecture()[0])
    return platform.architecture()[0] == xxbit
    # return xxbit == ('64bit' if sys.maxsize > 2**32 else '32bit')


def setup_default_pd_display(**kargs) -> None:  # pylint: disable=unused-argument
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    # pd.options.display.max_colwidth = -1
    pd.options.display.colheader_justify = "left"
    # pd.options.display.precision = 4


def trunc_year_by(series, divisor):
    return (series - series.mod(divisor)).astype(int)


def normalize_values(values: list[float]) -> list[float]:
    if len(values or []) == 0:
        return []
    max_value: float = max(values)
    if max_value == 0:
        return values
    values = [x / max_value for x in values]
    return values


def extract_counter_items_within_threshold(counter: dict[int, list[str]], low: int, high: int) -> set[str]:
    item_values: set[str] = set()
    for x, wl in counter.items():
        if low <= x <= high:
            item_values.update(wl)
    return item_values


def chunks(lst: list[Any], n: int | None) -> Iterable[Any]:

    if (n or 0) == 0:
        yield lst
    n = n or 1
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def plot_wordcloud(df_data: pd.DataFrame, token: str = "token", weight: str = "weight", **args):
    token_weights = dict({tuple(x) for x in df_data[[token, weight]].values})
    image = wordcloud.WordCloud(
        **args,
    )
    image.fit_words(token_weights)
    plt.figure(figsize=(12, 12))  # , dpi=100)
    plt.imshow(image, interpolation="bilinear")
    plt.axis("off")
    # plt.set_facecolor('w')
    # plt.tight_layout()
    plt.show()
