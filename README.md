# Name Transliteration

need to install fslite for epitran to work
<https://pypi.org/project/epitran/>

To do:

- implement extracting unicode ranges instead of removing emojis (<https://www.ling.upenn.edu/courses/Spring_2003/ling538/UnicodeRanges.html>)
- ~~make underscores turn into spaces in the cleansing stage~~
- ~~when a name is "LikeThis" separate into "Like This", ie. add spaces when appropriate~~
- remove words such as "Mr" and "Sir"
- get model to work properly lol

problems

- poetry added packages are not working as expected, is not found when run in poetry shell, work around is to directly call the virtual environment python on the python program I want to run

languages supported

- Chinese
- Spanish
- Arabic
- Japanese
- French

## Using the classes

### The filter class

```python
import name_transliteration.filtering as filter
# instantiate an instance of the class, when instantiating the language is also set
my_filter = filter.Filter("zh")

# to perform the filtering, we supply the path to where the gzip files are stored
my_filter.filterData("./../data/")

# save the filtered data as text for easy viewing
my_filter.saveDataAsText()
```

## The cleanse class

```python
import name_transliteration.cleansing as cleanse
# instantiate an instance of the class, when instantiating the data frame to be cleansed on is also set
my_cleanser = cleanse.Cleanse(my_filter.getDataFrame())

# perform the cleansing
my_cleanser.cleanseData()

# save the cleansed data as text for easy viewing, also in the format that can be processed by the model builder
my_cleanser.saveDataAsText()
```

## The model builder class

```python
import name_transliteration.model_trainer as model_trainer
# instantiate an instance of the class, when instantiating set the important variables of the class
model_trainer = model_trainer.ModelTrainer(data_path = './zh_language_cleansed.txt', num_samples = 650)

# run the whole training process
model_trainer.runWholeTrainProcess()

# test what it has learnt, not quite there yet :)
model_trainer.predict("dabudong")
```
