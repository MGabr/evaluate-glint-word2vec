# Glint-Word2Vec Evaluation

Scripts for comparing the default Spark ML implementation of Word2Vec with a custom 
implementation of Word2Vec using the Glint parameter server with custom specialized 
operations for efficient distributed Word2Vec computation.

## Usage

### Evaluation sets

The models are evaluated on the SimLex 999 and wordsim 353 evaluation sets.
The original english sets are provided as well as german sets from 
https://github.com/iraleviant/eval-multilingual-simlex.

Further word analogies to evaluate and visualize via PCA will need to be specified as csv file. 
An example is ``example_country_capitals_de.csv`` which consists of german country-capital 
relations. The country-capital relations from the original Word2Vec paper are also provided.

### Data sets

As dataset to train Word2Vec on you can download a wikipedia dump and then extract it 
to a text file with ``get_texts.py``. For testing, a subset of articles from a wikipedia 
dump can be created with ``get_analogy_texts.py``. Another possibility is getting 
specific wikipedia articles with ``get_articles_texts.py``. The articles to get will need
to be specified as txt file. An example is ``example_country_capitals_de_articles.txt``.

### Training and evaluating

For the training and evaluation you will need the ``glint-word2vec-assembly-1.0.jar`` 
and the python binding file ``ml_glintword2vec.py`` as zip from 
https://github.com/MGabr/glint-word2vec. These will have to be specified as ``--jars`` 
and ``--py-files`` options of ``spark-submit``.

``train_word2vec.py`` can then be used to train a Glint or standard Spark ML model and 
``evaluate_word2vec.py`` to evaluate and visualize word analogies using a model.

In most cases you will want to remove stop words before training a model to avoid 
exploding gradients. This can be done with the ``--stop-word-lang`` and 
``--stop-word-file`` options.

### Example

An example for evaluating Glint Word2Vec with default settings (150 partitions, 5 parameter servers)
on a german wikipedia dump on the original Word2Vec country-capitals analogies is the following.

```bash
python3 get_texts.py dewiki-latest-pages-articles.xml.bz2 dewiki-latest-pages-articles.txt
spark-submit --num-executors 5 --executor-cores 30 --jars glint-word2vec-assembly-1.0.jar --py-files ml_glintword2vec.zip train_word2vec.py dewiki-latest-pages-articles.txt dewiki-latest-pages-articles.model glint --stop-word-lang de --stop-word-file stopwords/dewiki.txt
spark-submit --num-executors 5 --executor-cores 1 --jars glint-word2vec-assembly-1.0.jar --py-files ml_glintword2vec.zip evaluate_word2vec.py evaluation/country_capitals_de.csv de dewiki-latest-pages-articles.model glint country_capitals_de.png
```

To evaluate Glint Word2Vec with parameter servers running in a separate Spark application you have to 
start them beforehand with a command like

```bash
spark-submit --num-executors 5 --executor-cores 20 --class glint.Main glint-word2vec-assembly-1.0.jar spark
```

Afterwards ``--parameter-server-host`` followed by the host of the parameter server master 
(visible in the output of the Spark application) has to be added as argument to ``train_word2vec.py``.
The separate parameter servers Spark application will also be terminated after training is finished.