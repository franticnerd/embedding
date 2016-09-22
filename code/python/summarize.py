import os

results = []
for f in os.listdir("./output"):
	result = []
	match_list = ['{','l ','t ','w ','mr:','mrr:','time:']
	for line in open("./output/"+f):
		line = line.strip()
		for s in match_list:
			if line.startswith(s):
				if s in ['mr:','mrr:']:
					result.insert(0,line)
				else:	
					result.append(line)
	if len(result)==len(match_list):
		results.append(result)
results.sort(reverse=True)
for result in results:
	print result