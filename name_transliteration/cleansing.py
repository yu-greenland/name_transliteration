import pandas as pd
import epitran
import editdistance
import pykakasi
import re
import os

"""
should never be used before the data has been filtered first
"""
class Cleanse:
    # the transliteration objects live here
    zh_translit = epitran.Epitran('cmn-Hans', cedict_file='cedict_ts.u8')
    es_translit = epitran.Epitran('spa-Latn')
    ar_translit = epitran.Epitran('ara-Arab')
    ja_translit = pykakasi.kakasi()
    en_translit = epitran.Epitran('eng-Latn')

    """
    can supply a dataframe and an edit threshold on the creation of the Cleanse class
    if a dataframe is supplied, the language should be automatically set
    """
    def __init__(self, language_dataframe:pd.DataFrame = None, edit_threshold = 8):
        if language_dataframe is None:
            self.language_dataframe = None
            self.language = None
            self.edit_threshold = edit_threshold
        else:
            self.language_dataframe = language_dataframe
            self.language = language_dataframe["language"].get(0)
            self.edit_threshold = edit_threshold
    
    """

    """
    def cleanseData(self):
        assert self.language_dataframe is not None, "language dataframe not yet defined, call the readData method before calling cleanseData"
        assert self.language is not None, "language not yet defined, call the readData method before calling cleanseData"

        # strip numbers
        self.language_dataframe['username'] = self.language_dataframe['username'].apply(lambda x: re.sub(r'\d+', '', x))
        self.language_dataframe['screen_name'] = self.language_dataframe['screen_name'].apply(lambda x: re.sub(r'\d+', '', x))

        # turn columns into series so we can enumerate through faster
        screen_name_series = self.language_dataframe['screen_name']
        username_series = self.language_dataframe['username']

        # a list containing the rows that are over the threshold
        rows_over_threshold = []

        # iterate over the name pairs and fill up the rows over threshold list
        for index, value in enumerate(screen_name_series):
            username = username_series[index]
            screen_name = value

            translit_username = self.translitUserName(username)
            translit_screen_name = self.translitScreenName(screen_name)

            # print(translit_username,translit_screen_name)

            # if the transliteration did nothing
            if translit_username == username:
                rows_over_threshold.append(index)
            elif translit_screen_name == screen_name:
                rows_over_threshold.append(index)
            else:
                # use edit distance with regards to string length
                edit_distance = editdistance.eval(translit_username, translit_screen_name)
                if edit_distance > self.edit_threshold:
                    rows_over_threshold.append(index)
        
        self.language_dataframe = self.language_dataframe.drop(rows_over_threshold)
        self.language_dataframe.reset_index(drop=True, inplace=True)
    
    """
    my hope with having a dedicated translit function for user names is that for different
    languages custom rules can be applied to them
    """
    def translitUserName(self, name:str) -> str:
        # if the name is not japanese we use IPA
        if self.language != 'ja':
            return self.en_translit.transliterate(name)
        else:
            # if the name is japanese we just return name
            # this is because we are not converting to IPA in the screen transliteration
            return name

    
    """
    my hope with having a dedicated translit function for screen names is that for different
    languages custom rules can be applied to them
    """
    def translitScreenName(self, name:str) -> str:
        # if the name is not japanese we use IPA
        if self.language != 'ja':
            if self.language == 'zh':
                return self.zh_translit.transliterate(name)
            elif self.language == 'es':
                return self.es_translit.transliterate(name)
            elif self.language == 'ar':
                return self.ar_translit.transliterate(name)
            else:
                print("language not supported")
                return ""
        else:
            # if the name is japanese we just return the romanised version
            translit = self.ja_translit.convert(name)
            roman = ''
            for item in translit:
                roman = roman + item['hepburn']
            return roman

    """
    read the dataframe from a file
    must not be empty and already been passed through the filter
    """
    def readData(self, file_path:str):
        self.language_dataframe = pd.read_json(file_path)
        self.language = language_dataframe["language"].get(0)
    
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
            self.language_dataframe.to_json(out_path + '/' +self.language+'_language_cleansed.json',orient="records")
        else:
            self.language_dataframe.to_json(out_path + '/' + file_name,orient="records")

    """
    set the edit threshold
    must be an integer
    """
    def setEditThreshold(self, edit_threshold:int):
        self.edit_threshold = edit_threshold
    
    """
    return the language data frame
    """
    def getDataFrame(self) -> pd.DataFrame:
        return self.language_dataframe