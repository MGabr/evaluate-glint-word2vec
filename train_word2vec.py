# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import time

from pyspark.ml.feature import Word2Vec, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql.types import Row


parser = ArgumentParser(description="Train a word2vec model.", formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("txtPath", help="The path of the text file to use for training")
parser.add_argument("modelPath", help="The path to save the trained model to")
parser.add_argument("modelType", help="The type of model to train", choices=("glint", "ml", "gensim"))

parser.add_argument("--step-size",
					help="The step size / learning rate. For glint model type too large values might result in "
						 "exploding gradients and NaN vectors", default=0.01875, type=float)
parser.add_argument("--vector-size", help="The vector size", default=100, type=int)
parser.add_argument("--window-size", help="The window size", default=5, type=int)

parser.add_argument("--num-partitions",
					help="The number of partitions. Should equal num-executors * executor-cores", default=150, type=int)

parser.add_argument("--stop-word-lang",
					help="The language to use for removing default stop words. "
						 "Empty string means no default stop word removal. ", default="")
parser.add_argument("--stop-word-file",
					help="The (additional) stop word file to use for removing stop words. "
						 "Empty string means no stop word removal with file. ", default="")

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


def train_spark():
	from ml_glintword2vec import ServerSideGlintWord2Vec

	# initialize spark session with required settings
	spark = SparkSession.builder \
		.appName("train word2vec") \
		.config("spark.driver.maxResultSize", "0") \
		.config("spark.kryoserializer.buffer.max", "2047m") \
		.config("spark.rpc.message.maxSize", "2047") \
		.config("spark.sql.catalogImplementation", "in-memory") \
		.config("spark.dynamicAllocation.enabled", "false") \
		.getOrCreate()

	sc = spark.sparkContext

	# choose model
	if args.modelType == "glint":
		word2vec = ServerSideGlintWord2Vec(
			seed=1,
			numPartitions=args.num_partitions,
			inputCol="sentence",
			outputCol="model",
			stepSize=args.step_size,
			vectorSize=args.vector_size,
			windowSize=args.window_size,
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
			vectorSize=args.vector_size,
			windowSize=args.window_size
		)

	# remove stop words and shuffle if specified
	if args.stop_word_lang or args.stop_word_file:
		sentences = sc.textFile(args.txtPath).map(lambda row: Row(sentence_raw=row.split(" "))).toDF()
		stopwords = []
		if args.stop_word_lang:
			stopwords += StopWordsRemover.loadDefaultStopWords(args.stop_word_lang)
		if args.stop_word_file:
			stopwords += sc.textFile(args.stop_word_file).collect()
		remover = StopWordsRemover(inputCol="sentence_raw", outputCol="sentence", stopWords=stopwords)
		sentences = remover.transform(sentences)
	else:
		sentences = sc.textFile(args.txtPath).map(lambda row: Row(sentence=row.split(" "))).toDF()

	# train and save model
	model = word2vec.fit(sentences)
	model.save(args.modelPath)

	# shutdown parameter server Spark application
	if args.modelType == "glint":
		model.stop(terminateOtherClients=True)

	# shutdown Spark application
	sc.stop()


def train_gensim():
	from gensim.corpora import TextCorpus
	from gensim.corpora.textcorpus import lower_to_unicode
	from gensim.models import Word2Vec as GensimWord2Vec

	start = time()

	stopwords = []
	if args.stop_word_lang:
		# starting spark only for this...
		spark = SparkSession.builder.appName("load stop words").getOrCreate()
		stopwords += StopWordsRemover.loadDefaultStopWords(args.stop_word_lang)
		spark.sparkContext.stop()
	if args.stop_word_file:
		with open(args.stop_word_file) as stop_word_file:
			stopwords += [word.strip("\n") for word in stop_word_file.readlines()]

	def remove_stopwords(tokens):
		return [token for token in tokens if token not in stopwords]

	corpus = TextCorpus(
		args.txtPath,
		dictionary={None: None},
		character_filters=[lower_to_unicode],
		token_filters=[remove_stopwords]
	)

	model = GensimWord2Vec(
		seed=1,
		alpha=args.step_size,
		size=args.vector_size,
		window=args.window_size,
		sample=1e-6,
        sg=1
	)
	model.build_vocab(corpus.get_texts())
	model.train(corpus.get_texts(), total_examples=model.corpus_count, epochs=model.epochs)
	model.save(args.modelPath)

	end = time()
	print("Gensim training took {} seconds".format(end - start))


if args.modelType == "glint" or args.modelType == "ml":
	train_spark()
else:
	train_gensim()
