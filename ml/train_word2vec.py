import sys
from argparse import ArgumentParser


parser = ArgumentParser(description="Train a word2vec model using ServerSideGlintWord2Vec.")
parser.add_argument("txtPath", help="The path of the text file to use for training")
parser.add_argument("modelPath", help="The path to save trained model")
args = parser.parse_args()


from pyspark.sql import SparkSession
from pyspark.sql.types import Row
from pyspark.ml.feature import Word2Vec


# initialize spark session with required settings
spark = SparkSession.builder \
	.appName("train word2vec") \
	.config("spark.driver.maxResultSize", "2g") \
	.getOrCreate()

sc = spark.sparkContext

# train word2vec model
sentences = sc.textFile(args.txtPath).map(lambda row: Row(sentence=row.split(" "))).toDF()
word2vec = Word2Vec(
	seed=1,
	numPartitions=50,
	inputCol="sentence",
	outputCol="model"
)
model = word2vec.fit(sentences)
model.save(args.modelPath)

