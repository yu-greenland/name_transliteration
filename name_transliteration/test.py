import name_transliteration.filtering as filter
import name_transliteration.cleansing as cleanse
import name_transliteration.model_trainer as model_trainer

import pandas as pd

my_filter = filter.Filter("ko")

my_filter.filterData("./../data_small/")              

my_filter.saveDataAsText()

# my_filter.saveData("chinese_data/")

# my_filter.saveData("chinese_data/", file_name="test.json")    

# print(my_filter.language_dataframe.head())

# df = pd.read_json('chinese_data/test.json')





# my_cleanser = cleanse.Cleanse(my_filter.getDataFrame())

# my_cleanser.cleanseData()

# print(my_cleanser.getDataFrame().head())

# my_cleanser.saveData("arabic_data/", file_name="test_cleansed.json")

# my_cleanser.saveDataAsText()





# model_trainer = model_trainer.ModelTrainer(data_path = './zh_language_cleansed.txt', num_samples = 650)

# model_trainer.runWholeTrainProcess()

# model_trainer.predict("dabudong")