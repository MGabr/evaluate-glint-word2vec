import sys
from argparse import ArgumentParser


parser = ArgumentParser(description="Train a word2vec model using ServerSideGlintWord2Vec.")
parser.add_argument("txtPath", help="The path of the text file to use for training")
parser.add_argument("modelPath", help="The path to save trained model")
parser.add_argument("driverHost", help="The IP address of the driver. Set to \"127.0.0.1\" for local testing")
parser.add_argument("unigramTableSize",
					help="The size of the unigram table. Set to a lower value if there is not enough memory locally",
					default=100000000)
args = parser.parse_args()


from pyspark.sql import SparkSession
from pyspark.sql.types import Row
from ml_glintword2vec import ServerSideGlintWord2Vec


# initialize spark session with required settings
spark = SparkSession.builder \
	.appName("train glint-word2vec") \
	.config("spark.driver.maxResultSize", "2g") \
	.getOrCreate()

sc = spark.sparkContext

# train word2vec model
sentences = sc.textFile(args.txtPath).map(lambda row: Row(sentence=row.split(" "))).toDF()
word2vec = ServerSideGlintWord2Vec(
	seed=1,
	numPartitions=50,
	inputCol="sentence",
	outputCol="model",
	numParameterServers=5,
	parameterServerMasterHost=args.driverHost,
	unigramTableSize=int(args.unigramTableSize)
)
model = word2vec.fit(sentences)
model.save(args.modelPath)

