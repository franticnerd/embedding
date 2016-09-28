import os
import ast

def get_ranked_results():
	results = []
	for f in os.listdir("./output"):
		result = []
		match_list = ['{','l ','t ','w ','mr:','mrr:','time:']
		complete = False
		for line in open("./output/"+f):
			line = line.strip()
			for s in match_list:
				if line.startswith(s):
					if s in ['mr:','mrr:']:
						result.insert(0,line)
						complete = True
					else:	
						result.append(line)
		if complete:
			results.append(result)
	results.sort(reverse=True)
	return results

def get_best_params():
	params_str = get_ranked_results()[0][2]
	return ast.literal_eval(params_str)

if __name__ == '__main__':
	for result in get_ranked_results():
		print result