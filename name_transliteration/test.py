import name_transliteration.filtering as filter

my_filter = filter.Filter("zh")

my_filter.filterData("./../data")

my_filter.saveData("chinese_data")