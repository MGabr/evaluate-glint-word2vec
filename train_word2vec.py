# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pyspark.ml.feature import Word2Vec, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql.types import Row


parser = ArgumentParser(description="Train a word2vec model.", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("txtPath", help="The path of the text file to use for training")
parser.add_argument("modelPath", help="The path to save the trained model to")
parser.add_argument("modelType", help="The type of model to train", choices=("glint", "ml"))
parser.add_argument("--num-partitions",
					help="The number of partitions. Should equal num-executors * executor-cores", default=150, type=int)
parser.add_argument("--step-size",
                    help="The step size / learning rate. For glint model type too large values might result in "
                         "exploding gradients and NaN vectors", default=0.01875, type=float)
parser.add_argument("--stop-word-lang", help="The language to use for removing default stop words. "
											  "Empty string means no default stop word removal", default="")
parser.add_argument("--stop-word-file", help="The (additional) stop word file to use for removing stop words. "
											 "Empty string means no stop word removal with file", default="")
parser.add_argument("--vector-size", help="The vector size", default=100, type=int)
parser.add_argument("--num-parameter-servers",
					help="The number of parameter servers to use. Set to 1 for local mode testing. "
						 "Only relevant for glint model type", default=5, type=int)
parser.add_argument("--parameter-server-host",
					help="The host master host of the running parameter servers. "
						 "If this is not set a standalone parameter server cluster is started in this Spark application. "
						 "Only relevant for glint model type", default="")
parser.add_argument("--batch-size",
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
	.config("spark.driver.maxResultSize", "8g") \
	.config("spark.kryoserializer.buffer.max", "2047m") \
	.config("spark.rpc.message.maxSize", "2047") \
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
		stepSize=args.step_size,
		vectorSize=args.vector_size,
		numParameterServers=args.num_parameter_servers,
		parameterServerHost=args.parameter_server_host,
		unigramTableSize=args.unigram_table_size,
		batchSize=args.batch_size
	)
else:
	word2vec = Word2Vec(
		seed=1,
		numPartitions=args.num_partitions,
		inputCol="sentence",
		outputCol="model",
		stepSize=args.step_size,
		vectorSize=args.vector_size
	)


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


model = word2vec.fit(sentences)
model.save(args.modelPath)


if args.modelType == "glint":
	model.stop(terminateOtherClients=True)

sc.stop()
