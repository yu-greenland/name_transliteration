import name_transliteration.filtering as filter
import name_transliteration.cleansing as cleanse

import pandas as pd

# my_filter = filter.Filter("zh")

# my_filter.filterData("./../data/")

# my_filter.saveData("chinese_data/")

# my_filter.saveData("chinese_data/", file_name="test.json")

# print(my_filter.language_dataframe.head())

df = pd.read_json('chinese_data/test.json')

my_cleanser = cleanse.Cleanse(language_dataframe=df)

my_cleanser.cleanseData()

print(my_cleanser.getDataFrame().head())