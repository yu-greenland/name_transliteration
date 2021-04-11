#!/usr/bin/env python
# coding: utf-8

# # Filtering
# 
# This note book will be dedicated to filtering the Twitter data.
# 
# This is part of the filtering stage described in the Design Document.
# 
# ## input:
#     - raw JSON twitter data
# ## output:
#     - a csv file with user name, screen name
#     
#     - (later on maybe include other meta data like verified or not and creation date)
#     
#     - names that are obviously wrong, such as names with just emojis will be filtered out
# 

import sys
import re
import gzip
import json
import os
import pandas as pd

# the language we want to capture
# language = sys.argv[1]
# print('language is: ' + language)



class filtering:
    def __init__(self, df_list, language):
        self.df_list = df_list
        self.language = language

    # reads in the data from specified folder, default is in ../data/
    # data should be in the form of gzipped files
    def readData(self, data_path="../data/"):
        # the path to the gzip twitter data
        data_path = data_path

        # the list that is going to contain all the dataframe information
        df_list = []

        # finds all files in the data path and combines them together
        for file in os.listdir(data_path):    
            if file.endswith(".gz"):        
                print(data_path+file)        
                with gzip.open(data_path+file) as f:
                    for line in f:
                        json_line = json.loads(line)
                        filtered_dict = {
                            "screen_name": json_line["user"]["screen_name"],
                            "username": json_line["user"]["name"],
                            "language": json_line["lang"]
                        }
                        df_list.append(filtered_dict)



# create the twitter data data frame
twitter_data = pd.DataFrame(df_list)

language_dataframe = twitter_data[twitter_data["language"] == self.language]



language_dataframe.head()



# remove emjojis

def deEmojify(text):
    regrex_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)
    return regrex_pattern.sub(r'',text)

language_dataframe["username"].apply(deEmojify)



# remove all rows where the username is empty, this removes names which are just emojis
language_dataframe = language_dataframe[language_dataframe.username != '']

language_dataframe.dropna(inplace=True)
language_dataframe.drop_duplicates()
language_dataframe.reset_index(drop=True, inplace=True)

print(language_dataframe.head())

# saves to disk
# try:
#     os.mkdir('filtered') 
# except OSError as error:
#     print('filtered folder already created')
# language_dataframe.to_json('filtered/'+language+'_language_filtered.json',orient="records",lines=True)
