## Name Transliteration

need to install fslite for epitran to work
https://pypi.org/project/epitran/

To do:

- implement extracting unicode ranges instead of removing emojis (https://www.ling.upenn.edu/courses/Spring_2003/ling538/UnicodeRanges.html)
- ~~ make filtering class and make it pretty ~~
- implement edit distance taking into account word lengths
- ~~ also make cleansing class and make it pretty ~~
- ~~ have special protocol for Japanese ~~

problems
- poetry added packages are not working as expected, is not found when run in poetry shell, work around is to directly call the virtual environment python on the python program I want to run
