# football

A project to scrape football data from the internet using Python, and then undertake data wrangling, feature selection, performance estimates (k-fold cross validation), machine learning evaluation, and machine learning prediction: 

## 1. scraping_football_data_v2.py
- Scrape premier league data from fbref using the Beautiful soup and read_html python libraries.
- Read in specific stats relating to shooting, passing, possession, defence and misc. 
- Concatenate all stats for each team together 

## 2. predicting_football_data_v1.py
- Data wrangling to reshape data into better format, and categeorise string variables (e.g., opponenet, venue)
- Feature selection to understand most important variables using chi-squared and extra trees classifiers. 
- Resample data to estimate how well algorithm will perform on unseen data, using logisitic regression model and k-fold cross validation 
- Test a selection of different ML algorithms (e.g. linear, deicison tree, ensemble, support vector machines, Naive bayes).
- Use the best performaing ML algorithm to read in future football fixtures and predict whether a team will win or lose. 
