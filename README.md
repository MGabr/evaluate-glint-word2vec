# Spark Word2Vec vs Glint Word2Vec

Scripts which compare the default Spark implementation of Word2Vec with a custom 
implementation of Word2Vec using the Glint parameter server with custom specialized 
operations for efficient distributed Word2Vec computation.

## How to run

Download a dump of the german wikipedia ``dewiki-latest-pages-articles.xml.bz2`` and extract 
it to a text file of sentences with 

```
python get_texts.py dewiki-latest-pages-articles.xml.bz2 dewiki-latest-pages-articles.txt
``` 

For local testing, you can create a subset of the german wikipedia consisting of 1000 
articles containing the visualized countries of capitals with

```
python get_country_capitals_texts.py
```

The following sections show how to test the Word2Vec implementations locally.\
For evaluation on a cluster with the whole text file of the dump,
this file will have to be put into HDFS and the paths and host parameter will have to be 
adjusted. See the help description of the scripts.


### Spark Word2Vec

Train Word2Vec with
```
spark-submit  ml/train_word2vec.py country_capitals.txt country_capitals.model
```

Create country/capitals visualization for Word2Vec with
```
spark-submit  ml/visualize_capitals_word2vec.py country_capitals.model country_capitals.png
```

### Glint Word2Vec

Build ``glint-word2vec-assembly-1.0.jar`` and get python binding file 
``ml_glintword2vec.py`` from https://github.com/MGabr/glint-word2vec.

Train GlintWord2Vec with
```
spark-submit  --jars glint-word2vec-assembly-1.0.jar --py-files ml_glintword2vec.py  glint/train_word2vec.py country_capitals.txt country_capitals_glint.model 127.0.0.1 1000000
```

Create country/capitals visualization for GlintWord2Vec with
```
spark-submit  --jars glint-word2vec-assembly-1.0.jar --py-files ml_glintword2vec.py  glint/visualize_capitals_word2vec.py country_capitals_glint.model country_capitals_glint.png
```
