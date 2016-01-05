import string
import csv

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

APP_NAME = "My Spark Application"

# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Function to break text into "tokens", lowercase them, remove punctuation and stopwords, and stem them
def tokenize(text):
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    #return no_stopwords
    return [w for w in stemmed if w]

def main(sc):	
	#text = sc.textFile("sample.txt")
    text = sc.textFile("WarAndPeace.txt")
    tokens = text.flatMap(lambda x: tokenize(x)) \
	    .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x, y: x + y ) \
	    .map(lambda x: (x[1], x[0])) \
	    .sortByKey(False)
    clean100 = tokens.take(100)
    with open("cl100.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerow(['Count', 'Word'])
	    for line in clean100:
		    writer.writerow(line)
    #wordCount = tokens.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b).map(lambda(x,y): (y,x)).sortByKey(ascending=False)
	#tokens.saveAsTextFile("clean_sort")
    wordcounts = text.flatMap(lambda x: x.split()) \
	    .map(lambda x: x.lower()) \
	    .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x,y: x+y) \
	    .map(lambda x: (x[1], x[0])) \
	    .sortByKey(False)
    top100 = wordcounts.take(100)
    with open("top100.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerow(['Count', 'Word'])
	    for line in top100:
		    writer.writerow(line)

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc)