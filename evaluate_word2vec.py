# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import codecs

import matplotlib

matplotlib.use('agg')

from matplotlib import pyplot
from numpy import dot, any
from numpy.linalg import norm
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lower, col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from sklearn.decomposition import PCA
from scipy.stats import spearmanr


def words_and_vecs_from_csv(spark, model, csv_filename):
    schema = StructType([StructField("word1", StringType(), False), StructField("word2", StringType(), False)])
    df = spark.read.csv(csv_filename, header=False, schema=schema)
    df = df.select(lower(col("word1")).alias("word1"), lower(col("word2")).alias("word2"))

    rows = df.collect()
    words1 = [row.word1 for row in rows]
    words2 = [row.word2 for row in rows]

    as_array = udf(lambda s: [s], ArrayType(StringType(), False))
    df = df.withColumn("word1", as_array("word1")).withColumn("word2", as_array("word2"))
    wordvecs1 = [row.model for row in model.transform(df.withColumnRenamed("word1", "sentence")).collect()]
    wordvecs2 = [row.model for row in model.transform(df.withColumnRenamed("word2", "sentence")).collect()]

    return words1, words2, wordvecs1, wordvecs2


def wordvecs_from_tsv(spark, model, tsv_filename):
    df = spark.read.option("sep", "\t").csv(tsv_filename, header=True)
    df = df.select(lower(col("word1")).alias("word1"), lower(col("word2")).alias("word2"))

    as_array = udf(lambda s: [s], ArrayType(StringType(), False))
    df = df.withColumn("word1", as_array("word1")).withColumn("word2", as_array("word2"))

    wordvecs1 = {row.sentence[0]: row.model for row
                 in model.transform(df.withColumnRenamed("word1", "sentence")).collect()}
    wordvecs2 = {row.sentence[0]: row.model for row
                 in model.transform(df.withColumnRenamed("word2", "sentence")).collect()}

    wordvecs = {}
    wordvecs.update(wordvecs1)
    wordvecs.update(wordvecs2)
    # remove words with zero vectors
    wordvecs = {word: vector for (word, vector) in wordvecs.items() if any(vector)}
    return wordvecs


def wordvecs_from_simlex(spark, model, language):
    return wordvecs_from_tsv(spark, model, "evaluation/simlex-" + language + ".txt")


def wordvecs_from_wordsim353(spark, model, language):
    return wordvecs_from_tsv(spark, model, "evaluation/wordsim353-" + language + ".txt")


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
        if words[0] == word2.encode("utf-8"):
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


def distance(v1, v2):
    """
    Returns the cosine distance between two vectors.
    """
    return 1 - dot(v1, v2) / ( norm(v1) * norm(v2) )


def correlation_and_coverage(word_vectors, language, source):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    """
    pair_list = []
    if source == "simlex":
        fread_simlex=codecs.open("evaluation/simlex-" + language + ".txt", 'r', 'utf-8')
    else:
        fread_simlex=codecs.open("evaluation/wordsim353-" + language + ".txt", 'r', 'utf-8')

    line_number = 0
    for line in fread_simlex:

        if line_number > 0:
            tokens = line.split()
            word_i = tokens[0].lower()
            word_j = tokens[1].lower()
            score = float(tokens[2])

            if word_i in word_vectors and word_j in word_vectors:
                pair_list.append( ((word_i, word_j), score) )
            else:
                pass
        line_number += 1

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    extracted_list = []
    extracted_scores = {}

    for (x,y) in pair_list:

        (word_i, word_j) = x
        current_distance = distance(word_vectors[word_i], word_vectors[word_j])
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)

    return round(spearman_rho[0], 3), coverage


def print_simlex_correlation(word_vectors, language):
    score, coverage = correlation_and_coverage(word_vectors, language, "simlex")
    print("SimLex-999 score and coverage: {}, {}".format(score, coverage))


def print_wordsim353_correlation(word_vectors, language):
    score, coverage = correlation_and_coverage(word_vectors, language, "wordsim353")
    print("WordSim overall score and coverage: {}, {}".format(score, coverage))


parser = ArgumentParser(description="Evaluate and visualize word analogies like countries to capitals of a ServerSideGlintWord2Vec model.")
parser.add_argument("csvPath", help="The path to the csv containing the word analogies to predict")
parser.add_argument("language", help="The language of the simlex and wordsim353 evaluation set to use", choices=("de", "en"))
parser.add_argument("modelPath", help="The path of the directory containing the trained model")
parser.add_argument("modelType", help="The type of model to train", choices=("glint", "ml"))
parser.add_argument("visualizationPath", help="The path to save the visualization to")
args = parser.parse_args()


from ml_glintword2vec import ServerSideGlintWord2VecModel


# initialize spark session with required settings
spark = SparkSession.builder \
    .appName("evaluate word2vec") \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.kryoserializer.buffer.max", "2047m") \
    .config("spark.rpc.message.maxSize", "2047") \
    .config("spark.sql.catalogImplementation", "in-memory") \
    .getOrCreate()

sc = spark.sparkContext


# load model
if args.modelType == "glint":
    model = ServerSideGlintWord2VecModel.load(args.modelPath)
else:
    model = Word2VecModel.load(args.modelPath)


# get required vectors with model on spark
words1, words2, wordvecs1, wordvecs2 = words_and_vecs_from_csv(spark, model, args.csvPath)
simlex_wordvecs = wordvecs_from_simlex(spark, model, args.language)
ws353_wordvecs = wordvecs_from_wordsim353(spark, model, args.language)
predicted_synonyms = word_synonyms(words1, model)
predicted_words2 = word_analogies(wordvecs1, wordvecs2)


# stop model and spark
if args.modelType == "glint":
    model.stop()

sc.stop()


# evaluate and print results
print_word_synonyms(words1, predicted_synonyms)
print_word_analogies(words1, words2, predicted_words2)
plot_word_analogies(words1, words2, wordvecs1, wordvecs2, save_plot_filename=args.visualizationPath)
print_simlex_correlation(simlex_wordvecs, args.language)
print_wordsim353_correlation(ws353_wordvecs, args.language)
