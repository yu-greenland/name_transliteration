import pandas as pd
import epitran
import editdistance
import pykakasi
import re
import os
import regex
import ko_pron

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
    fr_translit = epitran.Epitran('fra-Latn')


    """
    can supply a dataframe and an edit threshold on the creation of the Cleanse class
    if a dataframe is supplied, the language should be automatically set
    the default edit_threshold is 0.1, but it can be tuned and changed according to how strict you want the name pair similarities to be
    """
    def __init__(self, language_dataframe:pd.DataFrame = None, edit_threshold = 0.1):
        if language_dataframe is None:
            self.language_dataframe = None
            self.language = None
            self.edit_threshold = edit_threshold
        else:
            self.language_dataframe = language_dataframe
            self.language = language_dataframe["language"].get(0)
            self.edit_threshold = edit_threshold
    
    """
    applies transformations to the user name including
    - stripping numbers
    - replacing underscores with spaces
    - adding spaces between when we think a word ends
    """
    def transformUserName(self, line):
        # strip numbers
        text = re.sub(r'\d+', '', line)
        # underscores to spaces
        text = re.sub(r'_', ' ', text)
        # add a space between lower case and upper case words
        text = re.sub(r"(\w)([A-Z])", r"\1 \2", text)
        return text.lower().strip()

    """
    applies transformations to the user name including
    - stripping numbers
    - replacing underscores with spaces
    - adding spaces between when we think a word ends
    """
    def transformScreenName(self, line):
        # also remove any white space before and after word
        return line.lower().strip()

    """
    set verbose to True to print the name pairs and threshold that gets past the cleansing stage
    """
    def cleanseData(self, verbose=False):
        assert self.language_dataframe is not None, "language dataframe not yet defined, call the readData method before calling cleanseData"
        assert self.language is not None, "language not yet defined, call the readData method before calling cleanseData"

        # do transformations on username and screen name
        self.language_dataframe['username'] = self.language_dataframe['username'].apply(self.transformUserName)
        self.language_dataframe['screen_name'] = self.language_dataframe['screen_name'].apply(self.transformScreenName)

        # turn columns into series so we can enumerate through faster
        screen_name_series = self.language_dataframe['screen_name']
        username_series = self.language_dataframe['username']

        # a list containing the rows that are over the threshold
        rows_over_threshold = []

        # iterate over the name pairs and fill up the rows over threshold list
        for index, value in enumerate(screen_name_series):
            username = username_series[index]
            screen_name = value

            # print(username,screen_name)

            translit_username = self.translitUserName(username)
            translit_screen_name = self.translitScreenName(screen_name)

            # print(translit_username,translit_screen_name)

            # if the transliteration did nothing
            if translit_username == username and self.language != 'ja':
                # there is a problem here, if it is japenese this would always be the case
                rows_over_threshold.append(index)
            elif translit_screen_name == screen_name:
                rows_over_threshold.append(index)
                # pass
            else:
                # use edit distance with regards to string length
                edit_distance = self.evaluateEditDistance(translit_username, translit_screen_name)
                if edit_distance > self.edit_threshold:
                    rows_over_threshold.append(index)
                else:
                    if verbose:
                        print(username,screen_name)
                        print(translit_username,translit_screen_name)
                        print(edit_distance)

        self.language_dataframe = self.language_dataframe.drop(rows_over_threshold)
        self.language_dataframe.reset_index(drop=True, inplace=True)
    
    """
    if a name is very long, it is more likely to need more edits
    while if a name is very short, then the edit distance would be very small
    we want to treat short names and long names the same way
    in this method we use the average length of the two names and divide the edit distance by this
    by doing so, the threshold is normalised to be between 0 and 1 and also being below the threshold means more similar name pairs
    """
    def evaluateEditDistance(self, name1:str, name2:str):
        avg_length = (len(name1) + len(name2) / 2)
        return editdistance.eval(name1, name2) / avg_length
    
    """
    sometimes i also just want to see how the normal edit distance would behave
    """
    def normalEditDistance(self, name1:str, name2:str):
        # also have to set the edit threshold to this format scale
        self.edit_threshold = 5
        return editdistance.eval(name1, name2)

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
            elif self.language == 'fr':
                return self.fr_translit.transliterate(name)
            elif self.language == 'ko':
                # no clue what the second argument does... but it is needed
                return ko_pron.romanise(name, "mr")
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

    """
    saves language dataframe as text
    easier to load into keras this way
    """
    def saveDataAsText(self, out_path='./', file_name=None):
        print("Saving cleansed names. " + str(len(self.language_dataframe)) + " number of rows. ")
        just_names_df = self.language_dataframe[['username','screen_name']]
        if file_name is None:
            just_names_df.to_csv(out_path+self.language+'_language_cleansed.txt', header=None, index=None, sep='\t', mode='w')
        else:
            just_names_df.to_csv(out_path+file_name, header=None, index=None, sep='\t', mode='w')
    
    def getCleansedData(self):
        return self.language_dataframe[['username','screen_name']].to_numpy()
