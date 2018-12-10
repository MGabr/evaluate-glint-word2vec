import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row
from pyspark.ml.feature import Word2Vec

# input sentences hdfs filepath and output model hdfs filepath
inp, outp = sys.argv[1:3]

# initialize spark session with required settings
spark = SparkSession.builder \
	.appName("train word2vec") \
	.config("spark.driver.maxResultSize", "2g") \
	.getOrCreate()

sc = spark.sparkContext

# train word2vec model
sentences = sc.textFile(inp).map(lambda row: Row(sentence=row.split(" "))).toDF()
word2vec = Word2Vec(
	seed=1,
	numPartitions=50,
	inputCol="sentence",
	outputCol="model"
)
model = word2vec.fit(sentences)
model.save(outp)

