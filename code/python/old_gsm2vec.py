# below are obsolete
class Gsm2vec:
	def __init__(self):
		self.nt2vecs = None
		self.start_time = time.time()

	def train(self, nt2nodes, et2net, dim=100, sample_size=10000):
		# initialization not specified in the algorithm, got wrong at the beginning
		# self.nt2vecs = {nt:{node:np.random.rand(dim) for node in nt2nodes[nt]} for nt in ntList}
		self.nt2vecs = {nt:{node:(np.random.rand(dim)-0.5)/dim for node in nt2nodes[nt]} for nt in ntList}
		et2optimizer = {et:Optimizer(et,et2net[et],sample_size) for et in et2net}
		# for i in range(10):
		for i in range(sample_size):
			for et in et2net:
				et2optimizer[et].sgd_one_step(self.nt2vecs)
			if i%1000==0:
				print i, time.time()-self.start_time

class Optimizer:
	def __init__(self,et,net,sample_size,alpha=10,min_alpha=0.0001,nagative=5):
		self.tu = et[0]
		self.tv = et[1]
		self.net = net
		self.alpha = alpha
		self.min_alpha = min_alpha
		self.nagative = nagative
		# to be fixed: should consider the weight in computing the degree
		self.u2d = {u:len(net[u]) for u in net}
		self.u2samples = {u:iter(np.random.choice( net[u].keys(), self.u2d[u], 
							p=normed(net[u].values()) )) for u in net}
		self.sample_size = sample_size
		self.samples = iter(np.random.choice( self.u2d.keys(), sample_size, 
							p=normed(np.power(self.u2d.values(),4.0/3)) ))

	def sgd_one_step(self,nt2vecs):
		vecs_u,vecs_v = nt2vecs[self.tu],nt2vecs[self.tv]
		self.alpha -= (self.alpha - self.min_alpha) / self.sample_size
		u = self.samples.next()
		try:
			v = self.u2samples[u].next()
		except StopIteration:
			self.u2samples[u] = iter(np.random.choice(self.net[u].keys(), self.u2d[u], 
									p=normed(self.net[u].values()) ))
			v = self.u2samples[u].next()
		for j in range(self.nagative+1):
			if j==0:
				target = v
				label = 1
			else:
				target = np.random.choice(vecs_v.keys())
				if target in self.net[u]: continue
				label = 0
			f = np.dot(vecs_u[u],vecs_v[target])
			g = (label - expit(f)) * self.alpha
			# print (label - expit(f)), self.alpha, self.tu, self.tv
			vecs_u[u] += g*vecs_v[target]
			vecs_v[target] += g*vecs_u[u]