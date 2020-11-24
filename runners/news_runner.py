# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests
import os
import pandas as pd

r = requests.get('https://news.bbc.co.uk')

with open('data/bbc_' + str(pd.to_datetime('now')).replace(':', '_') + '.html', 'w', encoding="utf-8") as outfile:
    outfile.write(r.text)

