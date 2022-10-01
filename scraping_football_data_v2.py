#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:22:51 2022

@author: tom
"""

# -*- coding: utf-8 -*-
"""
Python script created by Tom Thorp 23/09/2022. 
This script reads in stats about a football league, and sorts them into panda dataframes,
saving them in csv output for further use in external scripts (e.g. plotting and ML). 
 
Below scraping method adapted from (link continues over 3 lines): https://github.com/oluwatosin17/
Web-Scraping-Football-Matches-From-The-EPL-With-Python-/blob/main/
Web%20Scraping%20Football%20Matches%20From%20The%20EPL%20.ipynb 
 
"""
# Importing libraries
import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup
import os



df = pd.read_html('https://fbref.com/en/comps/9/2021-2022/2021-2022-Premier-League-Stats')

for i in range(24):
 df[i].to_csv('/Users/tom/Documents/python/football/football/team_stats/'+np.str([i])+'.csv')
 

 