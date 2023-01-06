# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:00:49 2023

@author: tom
"""


"""

This script takes a csv file of information relating to football matches, formats the data into a more usable format, then uses
different ML appraoches to try to predict the next result for each team. 


"""

# Importing libraries
import pandas as pd
import numpy as np


#read in previously made csv file
matches = pd.read_csv(r"C:\Users\tt13\football\matches_20230106.csv", index_col=0)


"""

This next section drops the variables which will not be used and creates predictor variables which will be of use. 

"""

#Drop columns that are not needed
matches = matches.drop(columns=['Captain','Formation','Referee','Match Report', 'Notes'])

#convert date column into datetime 
matches["Date"] = pd.to_datetime(matches["Date"])

#convert the string venue variable into integer categories
matches["venue_code"] = matches["Venue"].astype("category").cat.codes










#Add in rolling stats to represent form (convert WLD to 3,0,1), gf, ga, poss




#Code relevant variables into numbers 








