import sys
from gensim.corpora import WikiCorpus

if __name__ == '__main__':

	inp, outp = sys.argv[1:3]

	with open(outp, 'w') as output:
		wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
		for text in wiki.get_texts():
			output.write(" ".join(text) + "\n")

