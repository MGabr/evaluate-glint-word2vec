import wikipedia

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(description="Get wikipedia articles as a txt file.",
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("articlesPath", help="The path of the txt file with wikipedia article titles")
parser.add_argument("outputPath", help="The output path")
parser.add_argument("--lang", help="The language of the wikipedia articles", default="en")
args = parser.parse_args()


wikipedia.set_lang(args.lang)
wikipedia.set_rate_limiting(True)
with open(args.outputPath, 'w') as output:
    with open(args.articlesPath, 'r') as articlesfile:
        for article in articlesfile:
            article = article.replace("\n", "").replace(" ", "_")

            output.write(wikipedia.page(article).content.lower())
