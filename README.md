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
- ~~chave a way of saving the model and loading it back up again so I don't have to re-train it~~
- make model give a couple of predictions instead of just one
- using trained model to evalutate on test data, to further filter out, split data into two and perform cross evaluation using trained models,  see if it improves , calculte probability of incorrect/correct pairs, like another metric to evaluate on

problems

- ~~poetry added packages are not working as expected, is not found when run in poetry shell, work around is to directly call the virtual environment python on the python program I want to run~~
- ~~japanese does not capture the Chōonpu, don't know why, manually added it into the regex~~

questions

- does the current workflow of going from raw Twitter data to model make sense? ie. are the inputs and outputs of each component like you imagined?
- how to get polyglot to work

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
model_trainer = model_trainer.ModelTrainer(data_path = './zh_language_cleansed.txt')

# run the whole training process, doing this will create a folder where the learnt weights go, this can be loaded up later
model_trainer.runWholeTrainProcess()

# test what it has learnt
model_trainer.predict("dabudong")

# plot the loss and accuracy
model_trainer.plotLoss()
model_trainer.plotAccuracy()
```

## Loading the pre-trained model and perform transliterations

```python
import name_transliteration.model_trainer as model_trainer

# we still have to load in the cleansed data into the model trainer
model_trainer = model_trainer.ModelTrainer(language='zh', data_path = './'+'zh'+'_language_cleansed.txt')

# this is because the decoder needs some information from the orignal data
# this shouldn't take too long
model_trainer.processData()

# replace 'zh_model_50' with whatever the model name was called when saving the data
# by default the model is saved as <language>_model_<number of epochs trained on>
model_trainer.createDecoderEncoder('zh_model_50')

# perform transliteration predictions
model_trainer.predict("johnathon")
```

## What story do I want to tell?

As outlined in the design document, the objective of this project is to build a system that can perform human-like non-professional name transliterations. These transliterations can be presented as an alternative to standard transliterations.

For the model to learn human-like non-professional transliterations of person names, the cleansing of names need to be lax enough to let through name transliterations that do not conform to standard transliterations but also harsh enough to not let through the name pairs that are not transliterations (random Twitter data).

To obtain this optimal cleansing point, we test over a range of edit-distances (the thing that controls how harsh/lax cleansing is) when the validation loss starts to really take a nose dive should be the optimal cleansing point. This is because if the model was learning name pairs that are not transliterations the validation loss would be significant and we want to have the model learn transliterations up to the point when the names are not transliterations.

We want to capture the Twitter data to the point where name pairs are not transliterations.

My hypothesis is that there is an "optimal" set of parameters in the data cleansing stage so that the system can perform human-like non-professional name transliterations. I define optimal cleansing as letting through all name pairs that can be regarded as a transliteration while not letting through name pairs that are in no way transliterations. A crude way of determining whether we have achieved this optimal cleansing is by taking a random sample of cleansed name pairs and manually assessing whether they are legitimate transliterations.
The primary parameter to vary is the edit threshold. The edit threshold controls how far from a standard transliteration a name pair can be before being filtered out. I expect that increasing the edit threshold will cause the validation loss to also increase. This is because as the name pairs start to become increasingly different from the standard transliteration we expect them to no longer be actual transliterations, the model will have no pattern to learn from. Having some validation loss is tolerable, but when the validation loss becomes drastically big we assume that this is when the model starts to learn from invalid transliteration pairs. At this point when the validation loss becomes drastically big should be where the optimal edit threshold is. Verification through manual inspection of the cleansed data can be performed.
