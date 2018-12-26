# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('agg')

import sys
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from matplotlib import pyplot


parser = ArgumentParser(description="Visualize a word2vec model via capital relations using ServerSideGlintWord2Vec.")
parser.add_argument("modelPath", help="The path of the directory containing the trained model")
parser.add_argument("visualizationPath", help="The path to save the visualization")
args = parser.parse_args()


from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import Word2VecModel


# initialize spark session with required settings
spark = SparkSession.builder \
        .appName("visualize capitals word2vec") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()


# words to plot
countries = [u"österreich", u"deutschland", u"frankreich", u"spanien", u"großbritannien", u"finnland", ]
capitals = [u"wien", u"berlin", u"paris", u"madrid", u"london", u"helsinki"]
words = countries + capitals


def plot_words(model, save_plot_filename=None):
	# get word vectors for words to plot
	words_df = spark.createDataFrame([Row(sentence=[word]) for word in words])
	vecs = [row.model for row in model.transform(words_df).collect()]

	# perform PCA
	pca = PCA(n_components=2)
	pca_vecs = pca.fit_transform(vecs)

	# plot words
	pyplot.scatter(pca_vecs[:, 0], pca_vecs[:, 1])

	for country, capital in zip(range(len(countries)), range(len(countries), 2 * len(countries))):
		pyplot.plot(pca_vecs[[country, capital], 0], pca_vecs[[country, capital], 1], linestyle="dashed", color="gray")

	for i, word in enumerate(words):
		pyplot.annotate(word, xy=(pca_vecs[i, 0], pca_vecs[i, 1]))

	# save or show plot
	if save_plot_filename:
		pyplot.savefig(save_plot_filename)
	else:
		pyplot.show()
	pyplot.clf()


model = Word2VecModel.load(args.modelPath)
plot_words(model, save_plot_filename=args.visualizationPath)

