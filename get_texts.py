from argparse import ArgumentParser

from gensim.corpora import WikiCorpus


parser = ArgumentParser(description="Convert a wikipedia dump into a txt file.")
parser.add_argument("wikiPath", help="The path of the wikipedia dump")
parser.add_argument("outputPath", help="The output path")
args = parser.parse_args()


with open(args.outputPath, 'w') as output:
	wiki = WikiCorpus(args.wikiPath, lemmatize=False, dictionary={})
	for text in wiki.get_texts():
		output.write(" ".join(text) + "\n")
