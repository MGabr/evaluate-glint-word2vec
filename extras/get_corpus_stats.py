# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from operator import add

from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql.types import Row


parser = ArgumentParser(description="Get the counts of the top words.", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("txtPath", help="The path of the text file")
parser.add_argument("--stop-word-lang", help="The language to use for removing stop words. "
                                             "Empty string means no stop word removal", default="")
parser.add_argument("--stop-word-file", help="The (additional) stop word file to use for removing stop words. "
                                             "Empty string means no stop word removal with file", default="")
parser.add_argument("--min-count", help="The minimum number of times a token must appear", default=5, type=int)
parser.add_argument("--top-count", help="The number of top words to get the frequency for", default=50, type=int)
args = parser.parse_args()


spark = SparkSession.builder \
    .appName("get frequent words") \
    .config("spark.sql.catalogImplementation", "in-memory") \
    .getOrCreate()
sc = spark.sparkContext


if args.stop_word_lang or args.stop_word_file:
    sentences = sc.textFile(args.txtPath).map(lambda row: Row(sentence_raw=row.split(" "))).toDF()
    stopWords = []
    if args.stop_word_lang:
        stopWords += StopWordsRemover.loadDefaultStopWords(args.stop_word_lang)
    if args.stop_word_file:
        stopWords += sc.textFile(args.stop_word_file).collect()
    remover = StopWordsRemover(inputCol="sentence_raw", outputCol="sentence", stopWords=stopWords)
    sentences = remover.transform(sentences)
else:
    sentences = sc.textFile(args.txtPath).map(lambda row: Row(sentence=row.split(" "))).toDF()


words = sentences.rdd.map(lambda row: row.sentence).flatMap(lambda x:  x)

wordCounts = words.map(lambda w: (w, 1)) \
    .reduceByKey(add) \
    .filter(lambda w: w[1] >= args.min_count) \

vocabCount = wordCounts.count()

wordCount = wordCounts.reduce(lambda x, y: ("", x[1] + y[1]))[1]

wordCounts = wordCounts.collect()
wordCounts.sort(key=lambda x: x[1], reverse=True)
topWordCounts = wordCounts[:args.top_count]


sc.stop()


print("Vocabulary size: " + str(vocabCount))
print("Words in corpus: " + str(wordCount))
for wordCount in topWordCounts:
    print(str(wordCount[0].encode("utf-8")) + ": " + str(wordCount[1]))
