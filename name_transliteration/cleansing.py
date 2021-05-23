import pandas as pd
import epitran
import editdistance
import pykakasi
import re
import os
import regex
import ko_pron
import numpy as np

class Cleanser:
    """
    The purpose of the cleanser class is to pick out legitimate name pairs.
    This is because the data we get from Twitter is usually very noisy (users do not have to enter corresponding name pairs).
    The cleansing of data is done by comparing edit distance of the english user name and the screen name that has undergone a standard transliteration.
    Should never be used on data that has passed through the filter class.
    Additionally, the cleanser class also has the mechanisms to create test sets.

    Represents a cleanser class, part of the overall name transliteration training pipeline
    """
    # the transliteration objects live here
    zh_translit = epitran.Epitran('cmn-Hans', cedict_file='cedict_ts.u8')
    es_translit = epitran.Epitran('spa-Latn')
    ar_translit = epitran.Epitran('ara-Arab')
    ja_translit = pykakasi.kakasi()
    en_translit = epitran.Epitran('eng-Latn')
    fr_translit = epitran.Epitran('fra-Latn')



    def __init__(self, initial_dataframe:pd.DataFrame = None, training_dataframe = None, testing_dataframe = None, edit_threshold = None):
        """
        Can supply a dataframe and an edit threshold on the creation of the Cleanse class.
        If a dataframe is supplied, the language should be automatically set.
        The default edit_threshold is not set. Should be set upon calling the cleanseData() method.
        """
        self.initial_dataframe = initial_dataframe
        self.training_dataframe = training_dataframe
        self.testing_dataframe = testing_dataframe
        self.testing_dataframe_cleanse_0 = None
        self.testing_dataframe_cleanse_0_1 = None
        self.testing_dataframe_cleanse_0_25 = None
        self.language = None
        self.edit_threshold = edit_threshold

        # used in the pseudo random number generator
        self.seed = 42

        # used to count how many lines have been read by the cleanser
        self.line_counter = 0
    
    def transformUserName(self, line):
        """
        Applies transformations to the user name including:
        - stripping numbers
        - replacing underscores with spaces
        - adding spaces between when we think a word ends
        - case folding
        - removing spaces before and after the name (this is an artifact of removing emojis in the filtering stage)
        These transformations are based upon observation of typical Twitter user names

        Parameters
        ----------
        line : str
            a name, or really any string

        Returns
        ----------
        transformed_name : str
            the name with transformations applied
        """
        # strip numbers
        text = re.sub(r'\d+', '', line)
        # underscores to spaces
        text = re.sub(r'_', ' ', text)
        # add a space between lower case and upper case words
        text = re.sub(r"(\w)([A-Z])", r"\1 \2", text)
        return text.lower().strip()


    def transformScreenName(self, line):
        """
        applies basic transformations to the screen name including
        - case folding
        - removing spaces before and after the name (this is an artifact of removing emojis in the filtering stage)

        Parameters
        ----------
        line : str
            a name, or really any string

        Returns
        ----------
        transformed_name : str
            the name with transformations applied
        """
        # also remove any white space before and after word
        return line.lower().strip()

    def cleanseData(self, data_frame:pd.DataFrame, edit_threshold, verbose=False):
        """
        Performs cleansing on a dataframe.
        Additionally, performs cleansing on name pairs.

        Parameters
        ----------
        data_frame : pd.DataFrame
            a name, or really any string
        
        edit_threshold : float
            the edit threshold (note we are using a modified edit-distance calculation where it ranges from 0 to 1) to cleanse using
        
        verbose : bool
            default is False, but if set to True, will print out the name pairs that pass cleansing along with the edit distance

        Returns
        ----------
        cleansed_df : pd.DataFrame
            a dataframe with rows that were above the edit-threshold removed
        """
        assert data_frame is not None, "language dataframe not yet defined, call the readData method before calling cleanseData"
        self.language = data_frame["language"].get(0)
        assert self.language is not None, "language not yet defined, call the readData method before calling cleanseData"

        self.edit_threshold = edit_threshold
        # data_frame.reset_index(inplace=True)

        # do transformations on username and screen name
        data_frame['username'] = data_frame['username'].apply(self.transformUserName)
        data_frame['screen_name'] = data_frame['screen_name'].apply(self.transformScreenName)

        # turn columns into series so we can enumerate through faster
        screen_name_series = data_frame['screen_name']
        username_series = data_frame['username']

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
                if edit_distance > edit_threshold:
                    rows_over_threshold.append(index)
                else:
                    if verbose:
                        print(username,screen_name)
                        print(translit_username,translit_screen_name)
                        print(edit_distance)
            self.line_counter = self.line_counter + 1

        cleansed_df = data_frame.drop(rows_over_threshold)
        cleansed_df.reset_index(drop=True, inplace=True)
        return cleansed_df
    
    def evaluateEditDistance(self, name1:str, name2:str) -> float:
        """
        A modified edit distance evaluation.
        If a name is very long, it is more likely to need more edits.
        While if a name is very short, then the edit distance would be very small (ie. only 1 or 2 edits).
        We want to treat short names and long names the same way.
        In this method we use the average length of the two names and divide the edit distance by this.
        By doing so, the threshold is normalised, being below the threshold means more similar name pairs.

        Parameters
        ----------
        name1 : str
            a name, or really any string
        
        name2 : str
            a name, or really any string

        Returns
        ----------
        edit_distance : float
            the modified edit distance between the two names
        """
        avg_length = (len(name1) + len(name2) / 2)
        return editdistance.eval(name1, name2) / avg_length
    
    def normalEditDistance(self, name1:str, name2:str) -> int:
        """
        The normal edit distance.

        Parameters
        ----------
        name1 : str
            a name, or really any string
        
        name2 : str
            a name, or really any string

        Returns
        ----------
        edit_distance : int
            the edit distance between the two names
        """
        # also have to set the edit threshold to this format scale
        self.edit_threshold = 5
        return editdistance.eval(name1, name2)

    def translitUserName(self, name:str) -> str:
        """
        Applied to user names, names that are in English.
        For languages that are not Japanese, we use IPA as an intermediary language to compare similarities. 
        For Japanese, we just return the name because we have a dedicated Japanese transliterator.

        Parameters
        ----------
        name : str
            a name, or really any string

        Returns
        ----------
        transliterated_name : str
            the name but transliterated to IPA, or not
        """
        # if the name is not japanese we use IPA
        if self.language != 'ja':
            return self.en_translit.transliterate(name)
        else:
            # if the name is japanese we just return name
            # this is because we are not converting to IPA in the screen transliteration
            return name

    def translitScreenName(self, name:str) -> str:
        """
        Applied to screen names, names that are not in English.
        For languages that are not Japanese, we transliterate to IPA as an intermediary language to compare similarities.
        For Japanese, we use a dedicated Japanese transliterator to transliterate name.

        Parameters
        ----------
        name : str
            a name, or really any string

        Returns
        ----------
        transliterated_name : str
            the name but transliterated to IPA, or Japanese
        """
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

    def setEditThreshold(self, edit_threshold):
        """
        Set the edit threshold.
        Must be a number.
        """
        self.edit_threshold = edit_threshold
    
    def getDataFrame(self) -> pd.DataFrame:
        """
        return the language data frame
        """
        return self.initial_dataframe
    
    def getTrainingDataFrame(self):
        return self.training_dataframe

    def getTestingDataFrame(self):
        return self.testing_dataframe

    def splitTrainTest(self, init_df, num_in_test_set = 5000):
        """
        Creates the train dataframe and test dataframes.
        This should be the first method called from the cleansing class. 
        Typically the dataframe from the filtering class is fed into this method.

        currently num_in_test_set is how many rows are reserved for test set before cleansing
        really it should be how many rows are given to test set, oh well

        setting num_in_test_set = 1000 generates around about
        30 for test set 1
        50 for test set 2
        100 for test set 3

        setting num_in_test_set = 2000 generates around about
        60 for test set 1
        80 for test set 2
        200 for test set 3

        setting num_in_test_set = 3000 generates around about
        100 for test set 1
        130 for test set 2
        300 for test set 3

        setting num_in_test_set = 4000 generates around about
        120 for test set 1
        170 for test set 2
        400 for test set 3

        setting num_in_test_set = 5000 generates around about
        160 for test set 1
        220 for test set 2
        500 for test set 3

        NOTE: these numbers are for japanese, cleansing on other languages will differ
        Parameters
        ----------
        init_df : pd.DataFrame
            the dataframe containing the data that is to be split into test and training data
        """
        # set the initial dataframe
        self.initial_dataframe = init_df

        # split into test
        prng = np.random.RandomState(self.seed)
        test_indices = prng.choice(len(self.initial_dataframe), size=num_in_test_set, replace=False)
        self.testing_dataframe = self.initial_dataframe.iloc[test_indices]
        self.testing_dataframe.reset_index(inplace=True)

        # training dataframe is everything else
        self.training_dataframe = self.initial_dataframe.loc[set(self.initial_dataframe.index) - set(test_indices)]
        self.training_dataframe.reset_index(inplace=True)


    def createTestDataSets(self):
        """
        Creates the three testing datasets
        This should be called after splitting into testing and training using splitTrainTest().

        The purpose of the test set is for the model to be evaluated on never before seen data.
        There are actually going to be three test sets produced
        - with edit threshold 0
        - with edit threshold 0.1 and below
        - with edit threshold 0.25 and below

        These three tests sets are going to be derived from the same initial dataset coming in from filtering.
        In this way, we can compare the same test data across different experiments such as those involving changing edit-threshold.
        """
        self.testing_dataframe_cleanse_0 = self.cleanseData(self.testing_dataframe, edit_threshold=0)
        self.testing_dataframe_cleanse_0_1 = self.cleanseData(self.testing_dataframe, edit_threshold=0.1)
        self.testing_dataframe_cleanse_0_25 = self.cleanseData(self.testing_dataframe, edit_threshold=0.25)

    def createTrainDataSet(self, edit_threshold):
        """
        This should be called after splitting into testing and training using splitTrainTest().
        This cleanses the training dataset using the supplied edit threshold.
        """
        self.edit_threshold = edit_threshold
        self.training_dataframe = self.cleanseData(self.training_dataframe, edit_threshold=edit_threshold)

    def saveTestAndTrain(self, out_path='./'):
        """
        Saves training data and the three test data files as text files.
        This is because it is easier to load into the model_trainer_and_tester class later on in the pipeline.
        Additionally, creates a cleansing statistics file and saves as a text file.

        Parameters
        ----------
        out_path : str
            default is current folder, can be set so that files are saved to a custom folder.
        """
        train_just_names_df = self.training_dataframe[['username','screen_name']]

        file_name = 'train'+'_'+str(int(self.edit_threshold*100))+'_edit_distance_language_cleansed.txt'
        train_just_names_df.to_csv(out_path+file_name, header=None, index=None, sep='\t', mode='w')
        
        test0_just_names_df = self.testing_dataframe_cleanse_0[['username','screen_name']]
        test0_1_just_names_df = self.testing_dataframe_cleanse_0_1[['username','screen_name']]
        test0_25_just_names_df = self.testing_dataframe_cleanse_0_25[['username','screen_name']]

        # test 1 has data that is cleansed with edit threshold 0
        file_name0 = 'test1_cleansed.txt'
        test0_just_names_df.to_csv(out_path+file_name0, header=None, index=None, sep='\t', mode='w')
        # test 2 has data that is cleansed with edit threshold 0.1
        file_name0_1 = 'test2_cleansed.txt'
        test0_1_just_names_df.to_csv(out_path+file_name0_1, header=None, index=None, sep='\t', mode='w')
        # test 3 has data that is cleansed with edit threshold 0.25
        file_name0_25 = 'test3_cleansed.txt'
        test0_25_just_names_df.to_csv(out_path+file_name0_25, header=None, index=None, sep='\t', mode='w')

        print("Saved cleansed names as: " + '\n' 
        + file_name + " " + str(len(self.training_dataframe)) + " number of rows. " + '\n'
        + file_name0 + " " + str(len(test0_just_names_df)) + " number of rows. " + '\n'
        + file_name0_1 + " " + str(len(test0_1_just_names_df)) + " number of rows. " + '\n'
        + file_name0_25 + " " + str(len(test0_25_just_names_df)) + " number of rows. " + '\n'
        )

        with open("cleansing_stats.txt", 'w') as f:
            f.write("language: " + self.language + '\n')
            f.write("training cleansed on edit threshold " + str(self.edit_threshold) + '\n')
            f.write(file_name + " " + str(len(self.training_dataframe)) + " number of rows. " + '\n')
            f.write(file_name0 + " " + str(len(test0_just_names_df)) + " number of rows. " + '\n')
            f.write(file_name0_1 + " " + str(len(test0_1_just_names_df)) + " number of rows. " + '\n')
            f.write(file_name0_25 + " " + str(len(test0_25_just_names_df)) + " number of rows. " + '\n')
            f.write("total number of lines read in: " + str(self.line_counter))