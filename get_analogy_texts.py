from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv

from gensim.corpora import WikiCorpus


parser = ArgumentParser(description="Get a number of articles containing analogy words from a wikipedia dump.",
						formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("csvPath", help="The path to the csv containing the word analogies which should be contained")
parser.add_argument("wikiPath", help="The path of the wikipedia dump")
parser.add_argument("outputPath", help="The output path")
parser.add_argument("--n-articles", help="The number of articles", type=int, default=1000)
args = parser.parse_args()


if args.wikiPath.endswith(".txt"):
	inp = open(args.wikiPath, "r")
	wiki_file = False
else:
	wiki = WikiCorpus(args.wikiPath, lemmatize=False, dictionary={})
	inp = wiki.get_texts()
	wiki_file = True

try:
	with open(args.csvPath) as csvfile:
		word_analogies = [row for row in csv.reader(csvfile, delimiter=",")]
		remaining_n_articles = args.n_articles
		with open(args.outputPath, "w") as out:
			for text in inp:
				for word_analogy in word_analogies:
					if word_analogy[0] in text and word_analogy[1] in text:
						if wiki_file:
							out.write(" ".join(text) + "\n")
						else:
							out.write(text)
						remaining_n_articles -= 1
						break
				if remaining_n_articles == 0:
					break
finally:
	inp.close()
