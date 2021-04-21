# coding: utf-8

import sys
import re
import gzip
import json
import os
import pandas as pd
import regex


class Filter:
    """
    Represents a filter class part of the overall name transliteration training pipeline


    language : str
        the language is what is going to be filtered upon, this must be in ISO 639-2 Language Code format
    
    contains the language we are filtering on
    contains the df which is just the raw data loaded in
    contains the language dataframe which is the filtered df

    """
    def __init__(self, language:str):
        self.language = language
        self.language_dataframe = None


    """
    top level function that performs the entire filtering process

    Parameters
    ----------
    data_path : str
        a string, the path to the folder containig the Twitter data

    Returns
    ----------
    language_dataframe : pd.DataFrame
        the DataFrame belonging to the class
        also sets the language_dataframe variable
    """
    def filterData(self, data_path:str):
        twitter_data_list = self.readData(data_path)
        twitter_data = pd.DataFrame(twitter_data_list)

        # filter down to specified language, the language must be in ISO 639-2 Language Code format
        df = twitter_data[twitter_data["language"] == self.language]

        # remove characters in screen name that do not belong to the unicode of the language
        df['screen_name'] = df['screen_name'].apply(self.removeNonLanguageCharacters)

        # remove all rows where the screen_name is empty, this removes names which do not have any characters in the unicode of the language
        df = df[df.screen_name != '']
        df.dropna(inplace=True)
        df.drop_duplicates()
        df.reset_index(drop=True, inplace=True)

        # assign the filtered data frame to the class level variable
        self.language_dataframe = df
        return self.language_dataframe

    """
    very exerimental right now, like everything here
    uses regex to only keep the unicode characters that belong to the language
    """
    def removeNonLanguageCharacters(self, line):
        clean_name = ""
        if self.language == 'zh':
            # join characters separated by space
            clean_name = " "
            match_list = regex.findall(r"(\p{Han})", line)
            clean_name = clean_name.join(match_list)
        elif self.language == 'es':
            match_list = regex.findall(r"(\p{Latin})", line)
            clean_name = clean_name.join(match_list)
        elif self.language == 'ar':
            match_list = regex.findall(r"(\p{Arabic})", line)
            clean_name = clean_name.join(match_list)
        elif self.language == 'ja':
            # join characters separated by space
            clean_name = " "
            match_list = regex.findall(r"(\p{Katakana}|\p{Hiragana}|\p{Han})", line)
            clean_name = clean_name.join(match_list)
        elif self.language == 'fr':
            match_list = regex.findall(r"(\p{Latin})", line)
            clean_name = clean_name.join(match_list)
        elif self.language == 'ko':
            # join characters separated by space
            clean_name = " "
            match_list = regex.findall(r"(\p{Hangul}|\p{Han})", line)
            clean_name = clean_name.join(match_list)
        else:
            print("language not supported")
        return clean_name

    """
    reads in data and additionally extracts the fields we are interested in
    extracts the screen name, user name and language

    data_path: a string, the path to the folder containig the Twitter data
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
                            "username": json_line["user"]["screen_name"],
                            "screen_name": json_line["user"]["name"],
                            "language": json_line["lang"]
                        }
                        df_list.append(filtered_dict)
        return df_list

    """
    saves the language dataframe as json
    creates the out_path folder if it does not exist

    has an optional argument to be able to have a custom file name
    """
    def saveData(self, out_path:str, file_name=None):
        try:
            os.mkdir(out_path)
        except OSError as error:
            print('folder already created')
        if file_name is None:
            self.language_dataframe.to_json(out_path + '/' +self.language+'_language_filtered.json',orient="records")
        else:
            self.language_dataframe.to_json(out_path + '/' + file_name,orient="records")
    
    """
    saves language dataframe as text
    easier to read and
    easier to load into keras this way
    """
    def saveDataAsText(self, out_path='./', file_name=None):
        just_names_df = self.language_dataframe[['username','screen_name']]
        if file_name is None:
            just_names_df.to_csv(out_path+self.language+'_language_filtered.txt', header=None, index=None, sep='\t', mode='w')
        else:
            just_names_df.to_csv(out_path+file_name, header=None, index=None, sep='\t', mode='w')

    """
    return the language data frame
    """
    def getDataFrame(self):
        return self.language_dataframe
    
    """
    sets the language data frame
    """
    def setDataFrame(self, df:pd.DataFrame):
        self.language_dataframe = df