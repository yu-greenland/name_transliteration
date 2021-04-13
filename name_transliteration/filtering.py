#!/usr/bin/env python
# coding: utf-8

import sys
import re
import gzip
import json
import os
import pandas as pd


"""

"""
class Filter:
    """
    when creating a new filter object, you first set the language you are filtering upon
    the language is a string specifying the language we want to filter on, note this must be in ISO 639-2 Language Code format
    this language will be used by the filter functions
    """
    def __init__(self, language:str):
        self.language = language
        self.language_dataframe = None

    """
    top level function that performs the entire filtering process
    
    data_path: a string, the path to the folder containig the Twitter data
    """
    def filterData(self, data_path:str):
        # read data from the data_path
        twitter_data_list = self.readData(data_path)
        twitter_data = pd.DataFrame(twitter_data_list)

        # filter down to specified language, the language must be in ISO 639-2 Language Code format
        df = twitter_data[twitter_data["language"] == self.language]

        # remove emojis
        df["username"].apply(self.deEmojify)
        # remove all rows where the username is empty, this removes names which are just emojis
        df = df[df.username != '']
        df.dropna(inplace=True)
        df.drop_duplicates()
        df.reset_index(drop=True, inplace=True)

        # assign the filtered data frame to the class level variable
        self.language_dataframe = df

    """
    reads in data and additionally extracts the fields we are interested in
    extracts the screen name, user name and language
    returns a list of dictionaries
    """
    def readData(self, data_path:str) -> list:
        # the list that is going to contain all the dataframe information
        df_list = []

        # finds all files in the data path and combines them together
        for file in os.listdir(data_path):    
            if file.endswith(".gz"):
                print(data_path+file)        
                with gzip.open(data_path+'/'+file) as f:
                    for line in f:
                        json_line = json.loads(line)
                        filtered_dict = {
                            "screen_name": json_line["user"]["screen_name"],
                            "username": json_line["user"]["name"],
                            "language": json_line["lang"]
                        }
                        df_list.append(filtered_dict)
        return df_list

    """

    """
    def deEmojify(self, text):
        regrex_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)
        return regrex_pattern.sub(r'',text)

    """

    """
    def saveData(self, out_path:str):
        try:
            os.mkdir(out_path)
        except OSError as error:
            print('filtered folder already created')
        self.language_dataframe.to_json(out_path + '/' +self.language+'_language_filtered.json',orient="records",lines=True)
