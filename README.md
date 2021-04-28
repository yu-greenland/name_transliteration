# Name Transliteration

need to install fslite for epitran to work
<https://pypi.org/project/epitran/>

To do:

- ~~implement extracting unicode ranges instead of removing emojis (<https://www.ling.upenn.edu/courses/Spring_2003/ling538/UnicodeRanges.html>)~~
- ~~make underscores turn into spaces in the cleansing stage~~
- ~~when a name is "LikeThis" separate into "Like This", ie. add spaces when appropriate~~
- remove words such as "Mr" and "Sir"
- ~~get model to work properly lol~~
- ~~currently when saving to text file, it appends to the end, make it not do that~~
- maybe look at substrings of screen name and user name to see if they can be a better match than using the whole screen name and user name
- from cleansing to model_trainer, I should be passing a tensor instead of saving a text file to disk
- automatically detect the size of the file passed into the model_trainer, using this automatically split data into testing and training, perhaps use sklearn train_test_split function
- create evaluation class
- make model give a couple of predictions instead of just one

problems

- ~~poetry added packages are not working as expected, is not found when run in poetry shell, work around is to directly call the virtual environment python on the python program I want to run~~
- ~~japanese does not capture the Chōonpu, don't know why, manually added it into the regex~~

questions

- does the current workflow of going from raw Twitter data to model make sense? ie. are the inputs and outputs of each component like you imagined?
- how to get polyglot to work
- i shouldn't need to have an attention mechanism in my model right? because i'm not translating sentences, i'm transliterating words

languages supported

- Chinese
- Spanish
- Arabic
- Japanese (does quite well tbh)
- French
- Korean

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

# test what it has learnt
model_trainer.predict("dabudong")
```
