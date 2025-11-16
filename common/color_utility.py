from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from bokeh.palettes import Category20_20, Set1_8, all_palettes  # pylint: disable=E0611

DEFAULT_ALL_PALETTES = all_palettes
DEFAULT_PALETTE = Category20_20
DEFAULT_LINE_PALETTE = Set1_8


class ColorGradient:

    @staticmethod
    def hex_to_RGB(rgb) -> list[int]:
        return [int(rgb[i : i + 2], 16) for i in range(1, 6, 2)]

    @staticmethod
    def RGB_to_hex(RGB) -> str:
        return "#" + "".join([f"0{v:x}" if v < 16 else f"{v:x}" for v in [int(x) for x in RGB]])

    @staticmethod
    def color_dict(gradient) -> dict[str, Any]:
        return {
            "hex": [ColorGradient.RGB_to_hex(RGB) for RGB in gradient],
            "r": [RGB[0] for RGB in gradient],
            "g": [RGB[1] for RGB in gradient],
            "b": [RGB[2] for RGB in gradient],
        }


class StaticColorMap:

    def __init__(self, palette: Sequence[str]) -> None:
        self.color_map: dict[Any, str] = {}
        self.palette: Sequence[str] = palette
        self.color_index: int = 0

    def next_color(self) -> str:
        self.color_index = (self.color_index + 1) % len(self.palette)
        return self.palette[self.color_index]

    def add_categories(self, categories: Sequence[str]) -> Self:
        unseen_categories: list[str] = list(set(categories) - set(self.color_map.keys() - {np.nan}))
        if len(unseen_categories) == 0:
            return self
        self.color_map.update({v: self.next_color() for v in unseen_categories})
        return self

    def get_palette(self, categories: Sequence[str]) -> list[str]:
        # add new categories
        self.add_categories(categories)
        return [self.color_map[k] for k in categories]


static_color_map = None


def get_static_color_map(
    palette: tuple[str, ...] | list[str] | None = None,
) -> StaticColorMap:
    global static_color_map  # pylint: disable=W0603
    palette = palette or DEFAULT_PALETTE  #  type: ignore
    if static_color_map is None:
        static_color_map = StaticColorMap(palette)
    return static_color_map
