# Spark Word2Vec vs Glint Word2Vec

Scripts for comparing the default Spark ML implementation of Word2Vec with a custom 
implementation of Word2Vec using the Glint parameter server with custom specialized 
operations for efficient distributed Word2Vec computation.

## How to run

The word analogies to evaluate will need to be specified as csv file. An example is 
``country_capitals.csv`` which consists of german country-capital relations.

As dataset to train Word2Vec on you can download a wikipedia dump and then extract it 
to a text file with ``get_texts.py``. For testing, a subset of articles from a wikipedia 
dump can be created with ``get_analogy_texts.py``. Another possibility is getting 
specific wikipedia articles with ``get_articles_texts.py``. The articles to get will need
to be specified as txt file. An example is ``country_capitals_articles.csv``.

For the training and evaluation you will need the ``glint-word2vec-assembly-1.0.jar`` 
and the python binding file ``ml_glintword2vec.py`` as zip from 
https://github.com/MGabr/glint-word2vec.

``train_word2vec.py`` can then be used to train a Glint or standard Spark ML model and 
``evaluate_word2vec.py`` to evaluate and visualize word analogies using a model.
