#!/bin/bash
find ./ -name '*.ipynb' -exec jupyter nbconvert --to script --stdout {} \; | grep -w import | sort | uniq
