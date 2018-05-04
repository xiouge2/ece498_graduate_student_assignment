import numpy as np
import random
from collections import Counter, OrderedDict
import networkx as nx
from math import log, factorial, exp
from scipy.special import perm

K = 40
# n_vertices: total num vertices
# k: num of groups
# g: group assignment of all vertices (np.array)
# n: num of vertices in each group (np.array)
# m: num of edges between two groups r and s (np.array 2D)

def init_groups(G, k, n_vertices, p):
	# assign group
	g = np.random.choice(k, n_vertices)
	# count number of element in each group
	n = np.zeros(K)
	for idx, count in Counter(g).items():
		n[idx] = count
	# set up edges between groups matrix
	m = count_m(G, k, g)
	# initialize log probability
	E = log_p(G, n, m, g, k, p)
	return g, n, m, E

def count_m(G, k, g):
	m = np.zeros((K, K))
	# iterate through all nodes
	for v in G:
		# iterate through all neighbors
		for neighbor in list(G.neighbors(v)):
			# add edge between groups
			m[g[v]][g[neighbor]]+=1
	return m

# n_vertices: total num vertices
# k: num of groups
# g: group assignment of all vertices (np.array)
# n: num of vertices in each group (np.array)
# m: num of edges between two groups r and s (np.array 2D)

# equation (4) multiply by degree-correction factor
def log_p(G, n, m, g, k, p):
	log_p_A_g = 0
	for r in range(k):
		log_p_A_g += log(factorial(n[r]))
		if n[r] > 0:
			# get the index of nodes in group r
			nodes_in_r = np.where(g==r)[0]
			# compute the sum of degree in group r
			kappa = sum(list(dict(G.degree(nodes_in_r)).values()))
			# correction factor
			# log((n-1)!/(n+k-1)!) = -log((n+k-1)!/(n-1)!) = -log(perm(n+k-1, k))
			log_p_A_g += kappa*log(n[r]) + log(factorial(n[r]-1)) - log(factorial(kappa+n[r]-1))
			# log_p_A_g += kappa*log(n[r]) - log(perm(int(kappa+n[r]-1), int(kappa), exact=True))
			# equation (4)
			log_p_A_g += log(factorial(m[r, r]/2)) - ((m[r, r]/2)+1)*log(0.5*p*(n[r]**2)+1)
			for s in range(r+1, k):
				try:
					log_p_A_g += log(factorial(m[r, s])) - (m[r, s]+1)*log(p*n[r]*n[s]+1)
				except:
					print(m.shape)
					print(r,s)	
	return log_p_A_g

def create_log_factorial_table(twom, n_vertices):
	length = twom + n_vertices + 1
	log_factorial = {}
	product = 1.0
	for n in range(1, length):
		product = product*n
		log_factorial[n] = product
	return log_factorial

def change_k(n, k, g, m, n_vertices, K):
	# decrease or increase k with equal probability
	if bool(random.getrandbits(1)):
		# get zero element indices
		zero_indices = np.nonzero(n == 0)[0]
		# count number of empty groups
		empty = len(zero_indices)
		# if there are any empty groups, remove one of them
		if empty>0:
			r = random.choice(zero_indices)
			# decrease k by 1
			k = k - 1
			for u in range(n_vertices):
				# the last group
				if g[u] == k:
					g[u] = r
			# set the vertices in the last group to r
			# g[np.where(g==k)[0]] = r
			# set the num of nodes in group k to be that of group r
			n[r] = n[k]
			# for s in range(k):
			# 	if r == s:
			# 		m[r][r] = m[k][k]
			# 	else:
			# 		m[r][s] = m[k][s]
			# 		m[s][r] = m[s][k]
			m[r, :] = m[k, :]
			m[:, r] = m[:, k]
			m[r, r] = m[k, k]
	else:
		# With probability k/(n+k) increase k by 1, adding an empty group
		if np.random.uniform(0, n_vertices+k) < k:
			if k < K:
				n[k] = 0
				for r in range(k+1):
					m[k][r] = 0
					m[r][k] = 0
				# m[k, :] = 0
				# m[:, k] = 0
				k = k + 1
	return k

def change_group():
	return None

def nmupdate(r, s, d, k, m, n):
	# move node from r to s
	n[r]-=1
	n[s]+=1
	for t in range(k):
		# deduct num of edges from r to all other groups
		m[r, t] -= d[t]
		# deduct num of edges from all other groups to r
		m[t, r] -= d[t]
		# add num of edges from s to all other groups
		m[s, t] += d[t]
		# deduct num of edges from all other groups to s
		m[t, s] += d[t]
	return m, n

def load_network():
	G=nx.karate_club_graph()
	return G

def sweep(n_vertices, g, G, k, E, p, m, n, K):
	new_E = np.zeros(K)
	boltzmann = np.zeros(K)
	accept = 0
	for i in range(n_vertices):
		# with probability q = 1/(n+1), do k update
		if np.random.uniform(0, n_vertices+1) < 1:
			k = change_k(n, k, g, m, n_vertices, K)
		# with probability 1-q, do group update
		else:
			# randomly select a node
			u = np.random.randint(0, n_vertices)
			# get the group r that node u belongs to 
			r = g[u]
			# initialize d, num of edges this node has to each group
			d = np.zeros(K)
			# iterate through the neighbor of node u
			for neighbor in list(G.neighbors(u)):
				# get the group that the neighbor belongs to, increment the count of edge to that group
				d[g[neighbor]] += 1

			# iterate through the groups
			for s in range(k):
				if s == r:
					# if the group itself, no need to switch group, just pass up E
					new_E[s] = E
				else:
					# otherwise, compute log probability if the node is moved from target group to other groups
					m, n = nmupdate(r, s, d, k, m, n)
					new_E[s] = log_p(G, n, m, g, k, p)
					# restore grouping
					m, n = nmupdate(s, r, d, k, m, n)
			boltzmann = np.exp(new_E-E)
			# normalizing factor
			Z = np.sum(boltzmann)

			x = np.random.uniform(0, Z)
			sum = 0
			for s in range(k):
				sum += boltzmann[s]
				if sum > x:
					break
			# change group
			if s != r:
				g[u] = s
				m, n = nmupdate(r, s, d, k, m, n)
				E = new_E[s]
				accept += 1
	return accept/n_vertices, k, E


def main():
	G = load_network()
	n_vertices = G.number_of_nodes()
	K = 40 # upper bound
	k = 40 # initial value
	twom = sum(dict(G.degree()).values())
	p = twom/(n_vertices**2)
	g, n, m, E = init_groups(G, k, n_vertices, p)
	MCSWEEPS = 10000
	for step in range(MCSWEEPS):
		proportion_change, k, E = sweep(n_vertices, g, G, k, E, p, m, n, K)
		print('k', k)
		print('E', E)
	# newE[s] = logp(n,m);
	# n_vertices = 10
	
	# init_groups(k, n_vertices)
	# n = np.array([0, 0, 0, 0, 1])
	# change_k(n)


if __name__ == '__main__':
	main()
