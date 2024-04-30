#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Python script created by Tom Thorp 04/09/2022. 

This script contains 4 functions that are used to scrape football data from the fbref website. These functions
are called in the runner.py script. The first function, get_team_urls, retrieves the URLs of all the teams in the
Premier League from the fbref website (The league can be changed by changing the URL in the function). The second 
function, get_squad_team, retrieves the squad codes and team names from the team URLs. The third function, get_team_data,
scrapes and retrieves user specified team data from the fbref website. The data is then stored in a pandas dataframe, 
and saved as a csv file in the assets folder. The fourth function, format_data, formats the data that has been scraped
into a more readable format, and keeps only the columns that are required for the model.

"""

# Importing required libraries
import pandas as pd
import numpy as np
import requests 
from bs4 import BeautifulSoup
import time 
from datetime import date

"""

The first 2 functions act as helper functions that are required in the third function to scrape the data.

"""

def get_team_urls():
    """
    Retrieves the URLs of all the teams in the Premier League from the fbref website.

    Returns:
        list: A list of URLs of all the teams in the Premier League.
    """
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    
    # Using requests to make html request 
    data = requests.get(url)
    # Convert html text into string
    soup = BeautifulSoup(data.text, 'html.parser')
    # Select the first table with the class "stats_table"
    standing_table = soup.select("table.stats_table")[0]
    # Find all the anchor tags within this table
    links  = standing_table.find_all("a")
    # Loop through links variable to get the href property from the anchor tag
    links = [l.get("href") for l in links]
    links = [l for l in links if '/squads/' in l]
    # Prepend website domain to the link to provide full link
    team_urls = [f"https://fbref.com{l}" for l in links]
    return team_urls


def get_squad_team(team_urls):
    """
    Get the squad team information from a list of team URLs.

    Args:
        team_urls (list): A list of URLs representing the team pages, output from get_team_urls().

    Returns:
        numpy.ndarray: A 2D array containing the squad codes and team names.
    """
    # Creating a list of squad codes
    squad_code = [url.split("/")[-2].replace("-Stats", "") for url in team_urls]
    # Create a list of team names
    team_name = [url.split("/")[-1].replace("-Stats", "") for url in team_urls]
    # Joining team name and codes to lookup on to create unique team url for data
    squad_team = np.column_stack((squad_code,team_name))
    return squad_team

"""

This following section uses a direct pd.read_html lookup onto the html and takes data from the 
shooting, passing, pass types, goal and shot creation, defensive action, possession and misc stats. 
This loops through each team, appending the results to the variable all_matches. This creates a list of 
dataframes. 

"""

def get_team_data(i, team_urls, squad_team, all_matches):
    """
    Scrapes and retrieves team data from football websites.

    Args:
        i (int): The index of the team in the `team_urls` and `squad_team` lists.
        team_urls (list): A list of URLs for each team.
        squad_team (list): A list of tuples containing squad information for each team.
        all_matches (list): A list to store the retrieved team data.

    Returns:
        None
    """
    data = requests.get(team_urls[i])
    #creating df of general stats from scores and fixtures tab
    matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
    stats_types = {
        'shooting': ["Date", "Sh", "SoT", "Dist", "G-xG"],
        'passing': ["Date", "xA", "KP", "1/3", "PPA", "PrgP"],
        'possession': ["Date", "Def Pen", "Att Pen", "Succ%"],
        'misc': ["Date", "CrdY", "CrdR", "Fls", "Fld", "Off", "TklW", "Recov"],
        'defense': ["Date", "Int", "Clr", "Err"]
    }
    for stats_type, columns in stats_types.items():
        stats_url = f"https://fbref.com/en/squads/{squad_team[i][0]}/2023-2024/matchlogs/all_comps/{stats_type}/{squad_team[i][1]}-Match-Logs-All-Competitions"
        temp = pd.read_html(stats_url)[0]
        temp.columns = temp.columns.droplevel()
        try:
            matches = matches.merge(temp[columns], on="Date")
        except ValueError:
            continue
    matches = matches[matches["Comp"] == "Premier League"]
    matches["Team"] = squad_team[i][1].replace("-", " ")
    all_matches.append(matches)
    # Print the team name after scraping the data with their league position which is index + 1
    print(f"{squad_team[i][1].replace('-', ' ')} scraped. League position: {i + 1}")
    time.sleep(2)


def export_data(all_matches):
    """
    Concatenates all the dataframes in the list and saves the dataframe to a CSV file with today's date.

    Args:
        all_matches (list): A list of dataframes containing the scraped data.

    Returns:
        None
    """
    match_df = pd.concat(all_matches)
    today = date.today().strftime("%Y-%m-%d")
    match_df.to_csv(f"assets/matches/matches_{today}.csv", index=False)
    print("Data exported successfully.")


def format_data(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the scraped data into a more readable format, drops unnecessary columns, 
    and creates more helpful variables like home/away venue, target variable, and result code.

    Args:
        matches (pandas.DataFrame): A DataFrame containing the scraped data.

    Returns:
        pandas.DataFrame: A DataFrame containing the formatted data.
    """
    return (matches.drop(columns=['Captain','Formation','Referee','Match Report', 'Notes'])
            .assign(Date = lambda df: pd.to_datetime(df["Date"]),
                    venue_code = lambda df: df["Venue"].astype("category").cat.codes,
                    opp_code = lambda df: df["Opponent"].astype("category").cat.codes,
                    target = lambda df: (df["Result"] == "W").astype("int"),
                    result_code = lambda df: df["Result"].astype("category").cat.codes))









 