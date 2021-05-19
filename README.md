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
- ~~chave a way of saving the model and loading it back up again so I don't have to re-train it~~
- make model give a couple of predictions instead of just one
- using trained model to evalutate on test data, to further filter out, split data into two and perform cross evaluation using trained models,  see if it improves , calculte probability of incorrect/correct pairs, like another metric to evaluate on
- ~~make train_test_split custom~~
- do data exploration beforehand get a grasp
- make the model use tensorflow data types
- use the shuffle thing and epoch thing

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
my_filter.filterData("./data/")

# alternatively we can specify the number of files to filter
# in this example only 5 twitter files will be read in
my_filter.filterData("./data/", 5)

# save the filtered data as text for easy viewing
my_filter.saveDataAsText()
```

## The cleanse class

```python
import name_transliteration.cleansing as cleanse

# instantiate the cleanser
my_cleanser = cleanse.Cleanser()

# given the DataFrame produced by the filter class, this method splits that DataFrame
# into a single train DataFrame and three test DataFrames
# these DataFrames reside inside the cleanser class
my_cleanser.splitTrainTest(my_filter.getDataFrame())

# this does the cleansing of the test datasets
# cleansing is pre-set as 0, 0.1 and 0.25 edit-threshold on the three test datasets
my_cleanser.createTestDataSets()

# this does the cleansing of the training dataset
# you can specify the edit_threshold to cleanse on
my_cleanser.createTrainDataSet(edit_threshold = 0.1)

# the model_trainer_and_tester class requires the data to be in text format
# this method saves the datasets of test and train as text files
my_cleanser.saveTestAndTrain()
```

## The model trainer and tester

```python
import name_transliteration.model_trainer as model_trainer_and_tester

# instantiate the model trainer and tester class
# have to provide the language and the number of epochs
trainer_and_tester = model_trainer_and_tester.ModelTrainerAndTester(
    language=language, 
    epochs=20
)

# instead of manually calling each individual method to build, compile and train the model
# this method chains all these methods together
# however, you have to provide the training text file, might change in the future
trainer_and_tester.runWholeTrainProcess('train_0_edit_distance_language_cleansed.txt')

# the training process takes a long time
# this plays an audio so you are alerted when the training process has finished
from IPython.display import Audio
sound_file = './sound/beep-03.wav'
Audio(sound_file, autoplay=True)

# to evaluate how well the model has learnt name transliterations
# we evaluate on unseen data, the three test datasets
# have to provide the model name that was created
# might change to having no required arguments since we have all the information already
trainer_and_tester.evaluateOnTestData("ja_model_20")

# saves stats
trainer_and_tester.saveTrainingStats()

# we have to save the dimensions of the model before we load it
```

## Loading the pre-trained model and perform transliterations

```python
import name_transliteration.model_trainer as model_trainer

# this has to 
```

## What story do I want to tell?

As outlined in the design document, the objective of this project is to build a system that can perform human-like non-professional name transliterations. These transliterations can be presented as an alternative to standard transliterations.

For the model to learn human-like non-professional transliterations of person names, the cleansing of names need to be lax enough to let through name transliterations that do not conform to standard transliterations but also harsh enough to not let through the name pairs that are not transliterations (random Twitter data).

To obtain this optimal cleansing point, we test over a range of edit-distances (the thing that controls how harsh/lax cleansing is) when the validation loss starts to really take a nose dive should be the optimal cleansing point. This is because if the model was learning name pairs that are not transliterations the validation loss would be significant and we want to have the model learn transliterations up to the point when the names are not transliterations.

We want to capture the Twitter data to the point where name pairs are not transliterations.

My hypothesis is that there is an "optimal" set of parameters in the data cleansing stage so that the system can perform human-like non-professional name transliterations. I define optimal cleansing as letting through all name pairs that can be regarded as a transliteration while not letting through name pairs that are in no way transliterations. A crude way of determining whether we have achieved this optimal cleansing is by taking a random sample of cleansed name pairs and manually assessing whether they are legitimate transliterations.
The primary parameter to vary is the edit threshold. The edit threshold controls how far from a standard transliteration a name pair can be before being filtered out. I expect that increasing the edit threshold will cause the validation loss to also increase. This is because as the name pairs start to become increasingly different from the standard transliteration we expect them to no longer be actual transliterations, the model will have no pattern to learn from. Having some validation loss is tolerable, but when the validation loss becomes drastically big we assume that this is when the model starts to learn from invalid transliteration pairs. At this point when the validation loss becomes drastically big should be where the optimal edit threshold is. Verification through manual inspection of the cleansed data can be performed.

Going back to the original goal of creating a transliteration system that can provide human-like non-professional transliterations for Japanese.
How to verify that the steps to create this transliteration system is legitimate?

An extension of this goal would be to generalize this over many languages.

I think filtering does what it's supposed to do and makes sense in the context that I am performing it in.

We go back to the very start of the system creation process of cleansing.

Using model to refine new model

1. Split initial filtered data into two sets: A and B
2. Using set A, run the cleansing and model training using the optimal edit-distance estimated.
3. From the model produced use it in place of the edit-distance cleansing technique. How to do this? Run predict on user name to acquire predicted name transliteration, if the predicted name is close enough to the screen name then accept it as cleansed.
4. Using these name pairs from this new cleansing technique, train new model
5. Compare loss and such on this new model
