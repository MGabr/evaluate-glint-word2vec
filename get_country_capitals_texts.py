

countries = ["österreich", "deutschland", "frankreich", "spanien", "großbritannien", "finnland", ]

capitals = {}
capitals["österreich"] = "wien"
capitals["deutschland"] = "berlin"
capitals["frankreich"] = "paris"
capitals["spanien"] = "madrid"
capitals["großbritannien"] = "london"
capitals["finnland"] = "helsinki"


limit = 1000
with open("country_capitals.txt", "w") as out:
	with open("dewiki-latest-pages-articles.txt", "r") as inp:
		for sentence in inp:
			for country in countries:
				if country in sentence and capitals[country] in sentence:
					out.write(sentence)
					limit -= 1
					break;
			if limit == 0:
				break;

