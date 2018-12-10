import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row
from ml_glintword2vec import ServerSideGlintWord2Vec

# input sentences hdfs filepath and output model hdfs filepath
inp, outp = sys.argv[1:3]

# initialize spark session with required settings
spark = SparkSession.builder \
	.appName("train word2vec") \
	.config("spark.driver.maxResultSize", "2g") \
	.getOrCreate()

sc = spark.sparkContext

# train word2vec model, takes too long - even with very small vector and window size
sentences = sc.textFile(inp).map(lambda row: Row(sentence=row.split(" "))).toDF()
word2vec = ServerSideGlintWord2Vec(
	seed=1,
	numPartitions=5,
	inputCol="sentence",
	outputCol="model"
)
model = word2vec.fit(sentences)
model.save(outp)

