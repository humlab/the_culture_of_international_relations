# -*- coding: utf-8 -*-
import os
from itertools import product, cycle, islice
import numpy as np

class ColorGradient:

    @staticmethod
    def hex_to_RGB(hex):
        return [ int(hex[i:i+2], 16) for i in range(1,6,2) ]

    @staticmethod
    def RGB_to_hex(RGB):
        return "#"+"".join(["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in [ int(x) for x in RGB ]])

    @staticmethod
    def color_dict(gradient):
        return {
            "hex": [ ColorGradient.RGB_to_hex(RGB) for RGB in gradient ],
            "r": [ RGB[0] for RGB in gradient ],
            "g": [ RGB[1] for RGB in gradient ],
            "b": [ RGB[2] for RGB in gradient ]
         }

    @staticmethod
    def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
        ''' returns a gradient list of (n) colors between two hex colors. start_hex and finish_hex should be the full
        six-digit color string, including the number sign ("#FFFFFF") '''
        # Starting and ending colors in RGB form
        s = ColorGradient.hex_to_RGB(start_hex)
        f = ColorGradient.hex_to_RGB(finish_hex)
        # Initilize a list of the output colors with the starting color
        RGB_list = [s]
        # Calcuate a color at each evenly spaced value of t from 1 to n
        for t in range(1, n):
            # Interpolate RGB vector for color at the current value of t
            curr_vector = [
                int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
                for j in range(3)
            ]
            # Add it to our list of output colors
            RGB_list.append(curr_vector)

        return ColorGradient.color_dict(RGB_list)

    @staticmethod
    def rand_hex_color(num=1):
        ''' Generate random hex colors, default is one, returning a string. If num is greater than 1, an array of strings is returned. '''
        colors = [
            RGB_to_hex([x*255 for x in rnd.rand(3)])
            for i in range(num)
        ]
        return colors[0] if num == 1 else colors


    @staticmethod
    def polylinear_gradient(colors, n):
        ''' returns a list of colors forming linear gradients between
          all sequential pairs of colors. "n" specifies the total
          number of desired output colors '''
        # The number of colors per individual linear gradient
        n_out = int(float(n) / (len(colors) - 1))
        # returns dictionary defined by color_dict()
        gradient_dict = ColorGradient.linear_gradient(colors[0], colors[1], n_out)

        if len(colors) > 1:
            for col in range(1, len(colors) - 1):
                next = ColorGradient.linear_gradient(colors[col], colors[col+1], n_out)
                for k in ("hex", "r", "g", "b"):
                    # Exclude first point to avoid duplicates
                    gradient_dict[k] += next[k][1:]

        return gradient_dict

class StaticColorMap():
    
    def __init__(self, palette):
        self.color_map = { }
        self.palette = palette
        self.color_index = 0
        
    def next_color(self):
        self.color_index = (self.color_index + 1) % len(self.palette)
        return self.palette[self.color_index]
        
    def add_categories(self, categories):
        categories = list(set(categories) - set(self.color_map.keys() - { np.nan }))
        if len(categories) == 0:
            return self
        self.color_map.update({ v: self.next_color() for v in categories })
        return self
    
    def add_categories2(self, categories):
        categories = list(set(categories) - set(self.color_map.keys() - { np.nan }))
        if len(categories) == 0:
            return self
        colors = list(islice(cycle(self.palette), None, len(categories)))
        self.color_map.update({ v: colors[i] for i, v in enumerate(categories) })
        return self
    
    def get_palette(self, categories):
        # add new categories
        self.add_categories(categories)
        return [ self.color_map[k] for k in categories ]