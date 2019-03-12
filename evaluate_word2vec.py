# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import matplotlib

matplotlib.use('agg')

from matplotlib import pyplot
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from sklearn.decomposition import PCA


def words_and_vecs_from_csv(spark, model, csv_filename):
    schema = StructType([StructField("word1", StringType(), False), StructField("word2", StringType(), False)])
    df = spark.read.csv(csv_filename, header=False, schema=schema)

    rows = df.collect()
    words1 = [row.word1 for row in rows]
    words2 = [row.word2 for row in rows]

    as_array = udf(lambda s: [s], ArrayType(StringType(), False))
    df = df.withColumn("word1", as_array("word1")).withColumn("word2", as_array("word2"))
    wordvecs1 = [row.model for row in model.transform(df.withColumnRenamed("word1", "sentence")).collect()]
    wordvecs2 = [row.model for row in model.transform(df.withColumnRenamed("word2", "sentence")).collect()]

    return words1, words2, wordvecs1, wordvecs2


def word_synonyms(words1, model):
    return [model.findSynonyms(word1, 5).head(5) for word1 in words1]


def print_word_synonyms(words1, predicted_synonyms):
    for predicted_synonym, word1 in zip(predicted_synonyms, words1):
        words = [ps.asDict()["word"].encode("utf-8") for ps in predicted_synonym]
        similarities = [round(ps.asDict()["similarity"], 4) for ps in predicted_synonym]
        print("Predicted synonyms {} for {} with similarity {}".format(words, word1.encode("utf-8"), similarities))


def word_analogies(wordvecs1, wordvecs2):
    word2_minus_word1_vec = wordvecs2[0] - wordvecs1[0]
    return [model.findSynonyms(word2_minus_word1_vec + wordvec1, 5).head(5) for wordvec1 in wordvecs1]


def print_word_analogies(words1, words2, predicted_words2):
    num_correct = 0
    for word1, word2, predicted_word2 in zip(words1, words2, predicted_words2):
        words = [pw.asDict()["word"].encode("utf-8") for pw in predicted_word2]
        similarities = [round(pw.asDict()["similarity"], 4) for pw in predicted_word2]
        print("Predicted analogies {} for {} with similarity {}".format(words, word1.encode("utf-8"), similarities))
        if words[0] == word2:
            num_correct += 1
    print("Predicted {} of {} analogies correctly".format(num_correct, len(words1)))


def plot_word_analogies(words1, words2, wordvecs1, wordvecs2, save_plot_filename=None):
    words = words1 + words2
    vecs = wordvecs1 + wordvecs2

    # perform PCA
    pca = PCA(n_components=2)
    pca_vecs = pca.fit_transform(vecs)

    # plot words
    pyplot.scatter(pca_vecs[:, 0], pca_vecs[:, 1])
    for word1, word2 in zip(range(len(words1)), range(len(words1), 2 * len(words1))):
        pyplot.plot(pca_vecs[[word1, word2], 0], pca_vecs[[word1, word2], 1], linestyle="dashed", color="gray")
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(pca_vecs[i, 0], pca_vecs[i, 1]))

    # save or show plot
    if save_plot_filename:
        pyplot.savefig(save_plot_filename)
    else:
        pyplot.show()
    pyplot.clf()


parser = ArgumentParser(description="Evaluate and visualize word analogies like countries to capitals of a ServerSideGlintWord2Vec model.")
parser.add_argument("csvPath", help="The path to the csv containing the word analogies to predict")
parser.add_argument("modelPath", help="The path of the directory containing the trained model")
parser.add_argument("modelType", help="The type of model to train", choices=("glint", "ml"))
parser.add_argument("visualizationPath", help="The path to save the visualization to")
args = parser.parse_args()


from ml_glintword2vec import ServerSideGlintWord2VecModel


# initialize spark session with required settings
spark = SparkSession.builder \
    .appName("evaluate word2vec") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.catalogImplementation", "in-memory") \
    .getOrCreate()

sc = spark.sparkContext


# load model and perform evaluation
if args.modelType == "glint":
    model = ServerSideGlintWord2VecModel.load(args.modelPath)
else:
    model = Word2VecModel.load(args.modelPath)

words1, words2, wordvecs1, wordvecs2 = words_and_vecs_from_csv(spark, model, args.csvPath)
predicted_synonyms = word_synonyms(words1, model)
predicted_words2 = word_analogies(wordvecs1, wordvecs2)

if args.modelType == "glint":
    model.stop()

sc.stop()

print_word_synonyms(words1, predicted_synonyms)
print_word_analogies(words1, words2, predicted_words2)
plot_word_analogies(words1, words2, wordvecs1, wordvecs2, save_plot_filename=args.visualizationPath)
