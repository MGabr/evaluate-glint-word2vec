# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql.types import Row


parser = ArgumentParser(description="Train a word2vec model.", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("txtPath", help="The path of the text file to use for training")
parser.add_argument("modelPath", help="The path to save the trained model to")
parser.add_argument("modelType", help="The type of model to train", choices=("glint", "ml"))
parser.add_argument("--num-partitions",
					help="The number of partitions. Should equal num-executors * executor-cores", default=150, type=int)
parser.add_argument("--num-parameter-servers",
					help="The number of parameter servers to use. Set to 1 for local mode testing. "
						 "Only relevant for glint model type", default=5, type=int)
parser.add_argument("--batchsize",
					help="The mini-batch size. Too large values might result in exploding gradients and NaN vectors. "
						 "Only relevant for glint model type", default=10, type=int)
parser.add_argument("--unigram-table-size",
					help="The size of the unigram table. Set to a lower value if there is not enough memory locally. "
						 "Only relevant for glint model type", default=100000000, type=int)
args = parser.parse_args()


from ml_glintword2vec import ServerSideGlintWord2Vec


# initialize spark session with required settings
spark = SparkSession.builder \
	.appName("train word2vec") \
	.config("spark.driver.maxResultSize", "2g") \
	.config("spark.sql.catalogImplementation", "in-memory") \
	.getOrCreate()

sc = spark.sparkContext


# train word2vec model
if args.modelType == "glint":
	word2vec = ServerSideGlintWord2Vec(
		seed=1,
		numPartitions=args.num_partitions,
		inputCol="sentence",
		outputCol="model",
		numParameterServers=args.num_parameter_servers,
		unigramTableSize=args.unigram_table_size,
		batchSize=args.batchsize
	)
else:
	word2vec = Word2Vec(
		seed=1,
		numPartitions=args.num_partitions,
		inputCol="sentence",
		outputCol="model"
	)

sentences = sc.textFile(args.txtPath).map(lambda row: Row(sentence=row.split(" "))).toDF()
model = word2vec.fit(sentences)
model.save(args.modelPath)


if args.modelType == "glint":
	model.stop()

sc.stop()
