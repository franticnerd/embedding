import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def dictCosine(dict1, dict2):
	dotProduct = 0.0
	module1 = 0.0
	module2 = 0.0
	for key in dict1.keys():
		module1 += dict1[key]*dict1[key]
		if (dict2.has_key(key)):
			dotProduct += dict1[key]*dict2[key]
	for key in dict2.keys():
		module2 += dict2[key]*dict2[key]
	if(module1==0 or module2==0):
		return 0
	return dotProduct/math.sqrt(module1*module2)

def listCosine(list1,list2):
	return cosine_similarity([list1],[list2])[0][0]

# a dict represent a distribution, where key is dimension and value is probability
def dictKLD(dict1, dict2):
	vectorizer = DictVectorizer()
	vec1,vec2 = vectorizer.fit_transform([dict1, dict2]).toarray()
	print entropy(vec1,vec2)
