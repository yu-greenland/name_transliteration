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
import name_transliteration.cleansing as cleanse
# instantiate an instance of the class, when instantiating the language is also set
my_filter = filter.Filter("zh")

# to perform the filtering, we supply the path to where the gzip files are stored
my_filter.filterData("./../data/")
```
