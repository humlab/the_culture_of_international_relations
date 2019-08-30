#!/bin/bash

# Shared link: https://drive.google.com/open?id=1V8KPeghLQ2iOMWkbPqff480zDSLa5YDX


# rm -f ./data/${FILENAME}

python3 ./google_drive.py 1V8KPeghLQ2iOMWkbPqff480zDSLa5YDX ./data/Treaties_Master_List.xlsx
python3 ./google_drive.py 1k4dOPuqR7oi4K8SazoGN6R40jOBWOdWp ./data/parties_curated.xlsx
python3 ./google_drive.py 19lEmVPu7hNmr1MaMpU0VvKL7muu-OKg9 ./data/country_continent.csv

