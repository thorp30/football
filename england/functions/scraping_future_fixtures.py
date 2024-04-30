#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python script created by Tom Thorp 02/06/2023

This script contains 3 functions to retrieve the future fixtures for the next week from the fbref website. The first
function, get_upcoming_fixtures, retrieves the upcoming fixtures for the Premier League from the fbref website in their raw form.
The second function, create_fixtures_df, creates a dataframe of fixtures with selected columns that are required for the model.
The third function, export_fixtures, exports the fixtures dataframe to a CSV file. 

"""

# Importing libraries
import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup
import time 
from datetime import date, timedelta

# Creating a dictionary class that will return a default value if the key is not found
class MissingDict(dict):
    def __missing__(self, key):
        return key

def get_upcoming_fixtures(days_ahead=2):
    """
    Retrieves the upcoming fixtures for the Premier League from a website.

    Parameters:
    - days_ahead (int): The number of days ahead to consider for upcoming fixtures. Default is 2.

    Returns:
    - pandas.DataFrame: A DataFrame containing the upcoming fixtures with columns 'Date' and other relevant information.
    """
    # Defining the url of the website, directing to league specific stats tab.
    url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    # Using pandas to read in the table from the website, and convert to df
    matches = pd.read_html(url, match="Scores & Fixtures")
    matches = pd.DataFrame(matches[0])
    # Convert the 'Date' column to datetime
    matches['Date'] = pd.to_datetime(matches['Date'])
    # We only want the future fixtures, so we will remove all games where the date is in the past.
    today = pd.to_datetime(date.today())
    # Removing all games before today's date
    matches = matches[matches["Date"] >= today]
    # We now want the games in the next week, so we will remove all games after 1 week from now.
    # Getting the date 1 week from now
    next_week = today + pd.DateOffset(days=days_ahead)
    # Removing all games after 1 week from now
    matches = matches[matches["Date"] <= next_week]
    return matches

"""

The function below creates a dataframe of fixtures with selected columns which is output from the get_upcoming_fixtures function.
The columns are "Date", "Team", "Opponent", "Venue", and "Target", where "Target" is the target variable for the model - in this case, "W" for win.

"""


def create_fixtures_df(matches):
    """
    Create a dataframe of fixtures with selected columns.

    Parameters:
    matches (DataFrame): A dataframe containing match data.

    Returns:
    DataFrame: A dataframe of fixtures with columns "Date", "Team", "Opponent", "Venue", and "Target".
    """
    # Creating a new df with the columns we want
    fixtures = pd.DataFrame(columns=["Date", "Team", "Opponent", "Venue", "Target"])
    # Creating a list of all the teams in the league taken from the matches df columns Home and Away.
    teams = pd.concat([matches["Home"], matches["Away"]])
    fixtures["Team"] = teams.values
    # Create a list of all opponents. This is taken from the matches df columns Home and Away.
    opponents = pd.concat([matches["Away"], matches["Home"]])
    fixtures["Opponent"] = opponents.values
    # Work out if venue is home or away.
    fixtures["Venue"] = np.where(fixtures["Team"].isin(matches["Home"]), "Home", "Away")
    # Get the date of the games from the matches df.
    fixtures["Date"] = fixtures.apply(lambda row: matches.loc[(matches["Home"] == row["Team"]) | (matches["Away"] == row["Team"]), "Date"].values[0], axis=1)
    # Reformat the date column to be in the format DD/MM/YYYY
    fixtures["Date"] = pd.to_datetime(fixtures["Date"]).dt.strftime("%d/%m/%Y")
    # Mapping dictionary to map team names to the names used in the model
    map_values = {
        "Brighton": "Brighton and Hove Albion",
        "Manchester Utd": "Manchester United",
        "Newcastle Utd": "Newcastle United",
        "Nott'ham Forest": "Nottingham Forest",
        "Tottenham": "Tottenham Hotspur",
        "West Ham": "West Ham United",
        "Wolves": "Wolverhampton Wanderers",
        "Sheffield Utd": "Sheffield United"
    }
    mapping = MissingDict(**map_values)
    # Mapping the opponent column using the mapping dictionary
    fixtures["Team"] = fixtures["Team"].map(mapping)
    # Assign W to target, as this will always be the target variable for the model
    fixtures["Target"] = "W"
    return fixtures


def export_fixtures(fixtures):
    """
    Export the fixtures dataframe to a CSV file with today's date in the filename.

    Parameters:
    fixtures (DataFrame): A dataframe containing the fixtures data.

    Returns:
    None
    """
    # Export the data to a csv file with today's date
    today = date.today().strftime("%Y-%m-%d")
    fixtures.to_csv(f"assets/fixtures/fixtures_{today}.csv", index=False)
    print("Fixtures exported successfully.")