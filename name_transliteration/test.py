import name_transliteration.filtering as filter
import name_transliteration.cleansing as cleanse

import pandas as pd

my_filter = filter.Filter("ar")

my_filter.filterData("./../data_small/")

# my_filter.saveData("chinese_data/")

# my_filter.saveData("chinese_data/", file_name="test.json")

# print(my_filter.language_dataframe.head())

# df = pd.read_json('chinese_data/test.json')

my_cleanser = cleanse.Cleanse(my_filter.getDataFrame())

my_cleanser.cleanseData()

# print(my_cleanser.getDataFrame().head())

my_cleanser.saveData("arabic_data/", file_name="test_cleansed.json")