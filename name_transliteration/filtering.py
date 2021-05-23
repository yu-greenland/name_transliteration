import gzip
import json
import os
import pandas as pd
import regex
pd.options.mode.chained_assignment = None

class Filter:
    """
    The purpose of the filter class is to read in raw Twitter blobs of data.
    These Twitter blobs are in JSON format.
    The filter class filters on language. All Twitter data that is not recognised as the chosen language is removed.
    The filtered data can be accessed by the DataFrame object that is created after filtering.

    Represents a filter class, part of the overall name transliteration training pipeline

    language : str
        the language is what is going to be filtered upon, this must be in ISO 639-2 Language Code format
    
    contains the language we are filtering on
    contains the language dataframe which is the filtered df

    """
    def __init__(self, language:str):
        self.language = language
        self.language_dataframe = None



    def filterData(self, data_path:str, num_files:int = None):
        """
        top level function that performs the entire filtering process

        Parameters
        ----------
        data_path : str
            a string, the path to the folder containig the Twitter data

        num_files : int
            the number of files that should be taken in, default it takes in all files in a folder

        Returns
        ----------
        language_dataframe : pd.DataFrame
            the DataFrame belonging to the class
            also sets the language_dataframe variable
        """
        twitter_data_list = self.readData(data_path, num_files=num_files)
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
        # return self.language_dataframe

    def removeNonLanguageCharacters(self, line:str) -> str:
        """
        Uses regex to only keep the unicode characters that belong to the language

        Parameters
        ----------
        line : str
            a name, or really any string

        Returns
        ----------
        clean_name : str
            the input string with all characters that are not part of the set language removed
        """
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
            match_list = regex.findall(r"(\p{Katakana}|\p{Hiragana}|\p{Han}|ー)", line)
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

    def readData(self, data_path, num_files:int=None) -> list:
        """
        Reads in data and additionally extracts the fields we are interested in.
        Extracts the screen name, user name and language.

        Parameters
        ----------
        data_path : str
            the data path to the folder that contains the Twitter blobs
        
        num_files : int (Optional)
            the number of files to be read in, if None all files found in the folder will be read in

        Returns
        ----------
        df_list : list
            a list of dictionaries containing name pairs
        """
        # the list that is going to contain all the dataframe information
        df_list = []

        # to keep track of how many files have been read in
        count = 1

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
            # stop taking in files if num_files is defined
            if num_files is not None:
                if count == num_files:
                    break
                count = count + 1
        return df_list

    def saveData(self, out_path, file_name=None):
        """
        saves the language dataframe as json
        creates the out_path folder if it does not exist

        has an optional argument to be able to have a custom file name
        """
        try:
            os.mkdir(out_path)
        except OSError as error:
            print('folder already created')
        if file_name is None:
            self.language_dataframe.to_json(out_path + '/' +self.language+'_language_filtered.json',orient="records")
        else:
            self.language_dataframe.to_json(out_path + '/' + file_name,orient="records")
    
    def saveDataAsText(self, out_path='./', file_name=None):
        """
        saves language dataframe as text
        easier to read
        """
        print("Saving filtered names. " + str(len(self.language_dataframe)) + " number of rows. ")
        just_names_df = self.language_dataframe[['username','screen_name']]
        if file_name is None:
            just_names_df.to_csv(out_path+self.language+'_language_filtered.txt', header=None, index=None, sep='\t', mode='w')
        else:
            just_names_df.to_csv(out_path+file_name, header=None, index=None, sep='\t', mode='w')

    def getDataFrame(self):
        """
        return the language data frame
        """
        return self.language_dataframe

    def setDataFrame(self, df):
        """
        sets the language data frame
        """
        self.language_dataframe = df