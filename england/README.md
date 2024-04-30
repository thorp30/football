# Predicting Premier League Football

A project to scrape football data from the internet using Python, and then undertake data wrangling, feature selection, performance estimates (k-fold cross validation) and machine learning evaluation, and machine learning prediction: 

## Runner.py  
- The controlling script for calling and using the functions to work through the entire process:
  - Data wrangling to reshape data into better format, and categeorise string variables (e.g., opponenet, venue)
  - Feature selection to understand most important variables using chi-squared and extra trees classifiers. 
  - Resample data to estimate how well algorithm will perform on unseen data, using logisitic regression model and k-fold cross validation 
  - Test a selection of different ML algorithms (e.g. linear, deicison tree, ensemble, support vector machines, Naive bayes).
  - Use the best performaing ML algorithm to read in future football fixtures and predict whether a team will win or lose. 
## assets
- Location of auto exported docs are stored (e.g., fixtures, matches, autoreport)

## docs 
- Location to store any useful documentation/ guides.

## functions 
- Location where all functions used and called from runner.py are stored. Functions are currently grouped together for relevance in the context of the process. 
