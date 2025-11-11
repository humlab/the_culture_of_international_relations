import random
from collections.abc import Sequence
from itertools import cycle, islice
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

    @staticmethod
    def linear_gradient(start_hex: str, finish_hex: str = "#FFFFFF", n: int = 10) -> dict[str, Any]:
        """returns a gradient list of (n) colors between two hex colors. start_hex and finish_hex should be the full
        six-digit color string, including the number sign ("#FFFFFF")"""
        # Starting and ending colors in RGB form
        s: list[int] = ColorGradient.hex_to_RGB(start_hex)
        f: list[int] = ColorGradient.hex_to_RGB(finish_hex)
        # Initilize a list of the output colors with the starting color
        RGB_list: list[list[int]] = [s]
        # Calcuate a color at each evenly spaced value of t from 1 to n
        for t in range(1, n):
            # Interpolate RGB vector for color at the current value of t
            curr_vector: list[int] = [int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)]
            # Add it to our list of output colors
            RGB_list.append(curr_vector)

        return ColorGradient.color_dict(RGB_list)

    @staticmethod
    def rand_hex_color(num: int = 1) -> str | list[str]:
        """Generate random hex colors, default is one, returning a string. If num is greater than 1, an array of strings is returned."""
        colors: list[str] = [
            ColorGradient.RGB_to_hex([random.random() * 255, random.random() * 255, random.random() * 255])
            for i in range(num)
        ]
        return colors[0] if num == 1 else colors

    @staticmethod
    def polylinear_gradient(colors: list[str], n: int) -> dict[str, Any]:
        """returns a list of colors forming linear gradients between
        all sequential pairs of colors. "n" specifies the total
        number of desired output colors"""
        # The number of colors per individual linear gradient
        n_out = int(float(n) / (len(colors) - 1))
        # returns dictionary defined by color_dict()
        gradient_dict: dict[str, Any] = ColorGradient.linear_gradient(colors[0], colors[1], n_out)

        if len(colors) > 1:
            for col in range(1, len(colors) - 1):
                v: dict[str, Any] = ColorGradient.linear_gradient(colors[col], colors[col + 1], n_out)
                for k in ("hex", "r", "g", "b"):
                    # Exclude first point to avoid duplicates
                    gradient_dict[k] += v[k][1:]

        return gradient_dict


class StaticColorMap:

    def __init__(self, palette: Sequence[str]) -> None:
        self.color_map: dict[Any, str] = {}
        self.palette: Sequence[str] = palette
        self.color_index: int = 0

    def next_color(self) -> str:
        self.color_index = (self.color_index + 1) % len(self.palette)
        return self.palette[self.color_index]

    def add_categories(self, categories: list[str]) -> Self:
        unseen_categories: list[str] = list(set(categories) - set(self.color_map.keys() - {np.nan}))
        if len(unseen_categories) == 0:
            return self
        self.color_map.update({v: self.next_color() for v in unseen_categories})
        return self

    def add_categories2(self, categories: list[str]) -> Self:
        categories = list(set(categories) - set(self.color_map.keys() - {np.nan}))
        if len(categories) == 0:
            return self
        colors: list[str] = list(islice(cycle(self.palette), None, len(categories)))
        self.color_map.update({v: colors[i] for i, v in enumerate(categories)})
        return self

    def get_palette(self, categories: list[str]) -> list[str]:
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
