import pandas as pd
import epitran
import editdistance

language = 'zh'

language_dataframe = pd.read_json(language+'_language_filtered.json', lines=True)

# do this for languages that have to be translated because it is very slow to do big translations
# language_dataframe = language_dataframe.loc[:100]

# print(language_dataframe.describe())

# strip numbers
import re
language_dataframe['username'] = language_dataframe['username'].apply(lambda x: re.sub(r'\d+', '', x))
language_dataframe['screen_name'] = language_dataframe['screen_name'].apply(lambda x: re.sub(r'\d+', '', x))

# turn columns into series so we can enumerate through faster
screen_name_series = language_dataframe['screen_name']
username_series = language_dataframe['username']

# global variables
edit_threshold = 8
rows_over_threshold = []
using_google_trans = False

# choosing which IPA translator to use
if language == 'zh':
    epi_username = epitran.Epitran('cmn-Hans', cedict_file='cedict_ts.u8')
elif language == 'es':
    epi_username = epitran.Epitran('spa-Latn')
elif language == 'ar':
    epi_username = epitran.Epitran('ara-Arab')
else:
    # if not found, we use google translate to translate to English
    # then use the english IPA translator
    print("using google translate")
    from googletrans import Translator
    translator = Translator()
    using_google_trans = True
    epi_username = epitran.Epitran('eng-Latn')

epi_screen_name = epitran.Epitran('eng-Latn')

# decide whether the screen name and user name are similar enough to keep
for index, value in enumerate(screen_name_series):
    username = username_series[index]
    screen_name = value

    if using_google_trans:
        translation = translator.translate(username, dest='en')
        username = translation.text
    ipa_username = epi_username.transliterate(username)
    ipa_screen_name = epi_screen_name.transliterate(screen_name)

    # if the transliteration did nothing
    if ipa_username == username:
        rows_over_threshold.append(index)
    elif ipa_screen_name == screen_name:
        rows_over_threshold.append(index)
    else:
        # use edit distance with regards to string length
        edit_distance = editdistance.eval(ipa_username, ipa_screen_name)
        if edit_distance > edit_threshold:
            rows_over_threshold.append(index)

language_dataframe = language_dataframe.drop(rows_over_threshold)
language_dataframe.reset_index(drop=True, inplace=True)
print(language_dataframe.describe())
print(language_dataframe.head())
language_dataframe.to_json(language+'_language_cleansed.json',orient="records",lines=True)