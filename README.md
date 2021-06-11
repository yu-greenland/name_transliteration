# Name Transliteration

Hello and welcome to my Advanced Topics in Computer Science project. This project is on using Twitter data to form a name transliteration model. The code in this repository make up the pipeline that turns Twitter data into a name transliteration model. This is a poetry project so in theory to install all necessary python libraries all you have to do is go ```poetry install```.

You need to install fslite for epitran to work.
<https://pypi.org/project/epitran/>

languages supported

- Japanese (main language used in demonstration and testing)
- Chinese
- Spanish
- Arabic
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
trainer_and_tester.runWholeTrainProcess('train_10_edit_distance_language_cleansed.txt', 'model_name')

# the training process takes a long time
# this plays an audio so you are alerted when the training process has finished
# not really necessary but I found it useful
from IPython.display import Audio
sound_file = './sound/beep-03.wav'
Audio(sound_file, autoplay=True)

# to evaluate how well the model has learnt name transliterations
# we evaluate on unseen data, the three test datasets
# have to provide the model name that was created
# might change to having no required arguments since we have all the information already
trainer_and_tester.evaluateOnTestData("model_name")

# saves stats
trainer_and_tester.saveTrainingStats()

# we have to save the dimensions of the model before we load it
```

## Loading the pre-trained model and perform transliterations

```python
import name_transliteration.model_trainer as model_trainer

loaded_model = model_trainer_and_tester.ModelTrainerAndTester(
    language='ja'
)
loaded_model.loadDataParameters()
loaded_model.createDecoderEncoder('model_A')
loaded_model.predict("yuzu")
```
