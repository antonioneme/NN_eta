"""
    edapfun.py implements the idea of inspecting high-dimensional datasets
    by means of the probability density function of the number of neighbors.
    Copyright (C) 2016  Antonio Neme Castillo.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>."

"""

import	sys, math, random, os, numpy, datetime
from	scipy import spatial
from	scipy import stats
import argparse
from	pyemd import emd


def	kullback_leibler_dsc(V1, V2):
	P = numpy.asarray(V1, dtype=numpy.float)
	Q = numpy.asarray(V2, dtype=numpy.float)
	cr = (P != 0.0) & (Q != 0.0)
	dd = numpy.sum(numpy.where(cr, Q * numpy.log(P / Q), 0))
	return dd

def	jensen_shannon(V1, V2):
	"""
	X = numpy.array(V1)
	Y = numpy.array(V2)
	D1 = X*numpy.log2(2*X/(X+Y))
	D2 = X*numpy.log2(2*Y/(X+Y))
	D1[numpy.isnan(D1)] = 0
	D2[numpy.isnan(D2)] = 0
	dd = 0.5*numpy.sum(D1+D2)    
	return dd
	"""
	"""
	X = V1
	Y = V2
	d1 = stats.entropy(X, Y, 2.0)
	d2 = stats.entropy(Y, X, 2.0)
	print "d = ", d1, d2
	dd = 0.5 * d1 + 0.5 * d2
	return dd
	"""
	d1 = kullback_leibler_dsc(V1, V2)
	d2 = kullback_leibler_dsc(V2, V1)
	dd = 0.5 * d1 + 0.5 * d2
	return dd
	

def	distance(V1, V2, Dd):
	dim = len(V1)
	dd = 0.0
	if Dd == 0:
		# Euclidean (L2)
		for i in range(dim):
			dd = dd + math.pow(V1[i] - V2[i], 2.0)
		dd = math.sqrt(dd)

	else:
		if Dd == 1:
			# Normal (L1)
			for i in range(dim):
				dd = dd + abs(V1[i] - V2[i])
		else:
			if Dd == 2:
				# Minkowski (Lp)
				for i in range(dim):
					dd = dd + math.pow(V1[i] - V2[i], 4.0)
				dd = math.pow(dd, 1.0/4.0)
			else:
				if Dd == 'sorensen':
					# Sorensen
					d1 = 0.0
					d2 = 0.0
					for i in range(dim):
						d1 = d1 + abs(V1[i] - V2[i])
						d2 = d2 + (V1[i] + V2[i])
					try:
						#dd = d1 / d2
						dd = abs(d1 / d2)
					except:
						dd = 0.0
				else:
					if Dd == 'jaccardx':
						# Jaccard
						d1 = 0.0
						d2 = 0.0
						d3 = 0.0
						d4 = 0.0
						for i in range(dim):
							d1 = d1 + (V1[i] * V2[i])
							d2 = d2 + (V1[i] * V1[i])
							d3 = d3 + (V2[i] * V2[i])
							d4 = d4 + (V1[i] * V2[i])
						if d2 + d3 - d4 != 0.0:
							dd = d1 / (d2 + d3 - d4)
						else:
							dd = 0.0
					else:
						if Dd == 5:
							# Cosine
							d1 = 0.0
							d2 = 0.0
							d3 = 0.0
							for i in range(dim):
								d1 = d1 + (V1[i] * V2[i])
								d2 = d2 + (V1[i] * V1[i])
								d3 = d3 + (V2[i] * V2[i])
							d2 = math.sqrt(d2)
							d3 = math.sqrt(d3)
							if (d2 * d3 != 0.0):
								dd = d1 / (d2 * d3)
							else:
								dd = 0.0
						else:
							# Fractional distance
							if Dd == 'fd':
								for i in range(dim):
									#print "V1 = ", V1[i], V2[i]
									dd = dd + math.pow(abs(V1[i] - V2[i]), 0.5)
									dd = math.sqrt(dd)
							else:
								if Dd == 'KL':
									# Kullback - Leibler
									dd = 0.0
									for i in range(dim):
										try:
											dd = dd + (V1[i] * math.log(V1[i] / V2[i]))
										except:
											dd = dd + 0.0
								else:
									if Dd == 'JS':
										#dd = jensen_shannon(V1, V2)
										#"""
										# Jensen - Shannon
										d1 = 0.0
										d2 = 0.0
										for i in range(dim):
											try:
												a1 = (2 * V1[i]) / (V1[i] + V2[i])
											except:
												a1 = 0.0
											if a1 != 0.0:
												try:
													d1 = d1 + ( V1[i] * math.log(a1) )
												except:
													d1 = d1 + 0.0
											try:
												b1 = (2 * V2[i]) / (V1[i] + V2[i])
											except:
												b1 = 0.0
											if b1 != 0.0:
												try:
													d2 = d2 + ( V2[i] * math.log(b1) )
												except:
													d2 = d2 + 0.0
										dd = abs(0.5 * (d1 + d2))
										#dd = 0.5 * (d1 + d2)
										#"""

	return dd

def	read_data(file):

	f = open(file, "r")
	x = f.readlines()
	f.close()

	Dats = []
	y = x[0].split('\t')
	dim = int(len(y))
	print "dim = ", dim
	for i in range(0, len(x)):
		xx = x[i].split('\t')
		tmp = []
		for j in range(dim):
			tmp.append(float(xx[j]))
		Dats.append(tmp)

	return [Dats, len(x), dim]
	#return [Dats, len(x) - 1, dim]

def	isNaN(A):
	return A != A

"""
n: number of vectors
DM2: the matrix computed by scipy.
mx: the diameter of the space.
rad:	the method for radii selection
nR:	number of radii
rearrange arranges the distance matrix so the original version of edapfun can
compute the histograms.

Returns:
[Distance matrix in new format, Histogram of distances (integer distances)]
"""
def	rearrange(n, DM2, mn, mx, rad, nR):
	Dt = [None] * n
	for i in range(n):
		Dt[i] = [0.0] * n
	s = 0
	# Histogram of distances
	print "rad = ", rad
	if rad == 0 or rad == 2:
		PD = []
	else:
		if rad == 1:
			# The radii with higher densities of neighbors should
			# be splitted more accurately.
			#PD = [0.0] * (int(mx) + 1)
			PD = [0.0] * (nR + 1)
			k_means = cluster.KMeans(n_clusters=nR, n_init = 10, max_iter = 1200)
			Centres = []
			for i in range(len(k_means.cluster_centers_)):
				Centres.append([k_means.cluster_centers_[i], i])
			Centres_sort = sorted(Centres)
			# Pending....

	for i in range(n):
		for j in range(i + 1, n):
			if i != j:
				if isNaN(DM2[s]):
					DM2[s] = 0.0
				Dt[i][j] = DM2[s]
				Dt[j][i] = DM2[s]
				#print "s = ", s, i, j, DM2[s]
				s = s + 1
				dd = int(Dt[i][j])
				# discretize 
				if rad == 1:
					qdd = discretize(qdd, mx, nR)
					PD[qdd] = PD[qdd] + 1.0
					# Pending ...
	return [Dt, PD]


def	which_metric(Dd):
	if Dd == 0:
		m = 'euclidean'
	else:
		if Dd == 1:
			m = 'cityblock'
		else:
			if Dd == 2:
				m = 'minkowski'
			else:
				if Dd == 4:
					m = 'jaccardx'
				else:
					if Dd == 5:
						m = 'cosine'

	return m

def	is_in(a, L):
	for i in L:
		if a == i:
			return 1
	return -1
"""
numvect: number of vectors (datapoints)
Dats: The data points
Dd: the metric
distance_matrix computes the distance matrix of vectors in Dats, considering
Dd as the metric. Also, it obtains some properties of the data based on
the distance matrix
nR: number of radii

Returns:
[Distance matrix, average distance over all pairs, maximum distacne (diameter),
minimum distance, the ratio of the probability (frequency) of consecutive
distances (sorted), histogram of distances]

"""
def	compute_distance_matrix(numvect, Dats, metric, rad, nR):
	# Compute the distance matrix
	if is_in(metric, ['euclidean', 'cityblock', 'minkowski', 'cosine', 'correlation', 'chebyshev', 'mahalanobis', 'canberra', 'jaccard']) == 1:
	#if metric == 'euclidean' or metric == 'cityblock' or metric == 'minkowski' or metric == 'cosine':
		# The metrics
		Dx = numpy.asarray(Dats)
		#metric = which_metric(Dd)
		h1 = datetime.datetime.now()
		print "metric = ", metric
		DM = spatial.distance.pdist(Dx, metric)
		h2 = datetime.datetime.now()
		print "Computing the distance matrix took... "
		print "h1 = ", h1
		print "h2 = ", h2
		max_dist = max(DM)
		min_dist = min(DM)
		if isNaN(min_dist):
			min_dist = 0.0
		print "rearrangement of matrix"
		DMx = spatial.distance.squareform(DM)
		PD = []
		max_dist = max(DM)
		min_dist = min(DM)
		"""
		if isNaN(max_dist):
			max_dist = 0.0
			for kk in range(len(DM)):
				if not isNaN(DM[kk]) and DM[kk] > max_dist:
					max_dist = DM[kk]
		print "md = ", max_dist
		[DMx, PD] = rearrange(numvect, DM, min_dist, max_dist, rad, nR)
		"""
		print "xxx"
		PDrat = []
		lp = len(PD)
		print "lp = ", lp
		for i in range(1, len(PD)):
			print "i = ", i
			if PD[i-1] > 0:
				rat = PD[i] / PD[i-1]
			else:
				rat = PD[i]
			PDrat.append([rat, i])
		PD_S = sorted(PDrat, reverse=True)

		h3 = datetime.datetime.now()
		print "h3 = ", h3
		print "rearrangement done!"
		avg_dist = 0.0
		for j in range(len(DM)):
			avg_dist = avg_dist + DM[j]
		avg_dist = avg_dist / float(len(DM))
		#avg_dist = avg_dist / float(numvect*(numvect - 1))
		#return [DMx, avg_dist, max_dist, min_dist, PD_S[0:30], PD]
		return [DMx, avg_dist, max_dist, min_dist, PD_S, PD]
	else:
		DIST = [None] * numvect
		for i in range(numvect):
			if i % 50 == 0:
				print "creating vector ", i
			DIST[i] = [0.0] * numvect
		max_dist = 0.0
		min_dist = 100000000.0
		avg_dist = 0.0
		t = 0
		PD = []
		for i in range(numvect):
			if i % 50 == 0:
				print "Processing distance for vector ", i
			# The Shannon's family of pseudometrics ...
			for j in range(numvect):
				if i != j:
					DIST[i][j] = distance(Dats[i], Dats[j], metric)
					PD.append(DIST[i][j])
					#print "D = ", DIST[i][j]
					avg_dist = avg_dist + DIST[i][j]
					if DIST[i][j] < min_dist and DIST[i][j] > 0.0:
						min_dist = DIST[i][j]
				t = t + 1

			mx_dist = max(DIST[i])
			if mx_dist > max_dist:
				max_dist = mx_dist

		avg_dist = avg_dist / t
		#print "max dist = ", max_dist
		#print "min dist = ", min_dist
		#print "avg dist = ", avg_dist
		PDrat = []
		for i in range(1, len(PD)):
			if PD[i-1] > 0:
				rat = PD[i] / PD[i-1]
			else:
				rat = PD[i]
			PDrat.append([rat, i])
		PD_S = sorted(PDrat, reverse=True)

		return [DIST, avg_dist, max_dist, min_dist, PD_S, PD]

def	read_distance_matrix(file):
	cont = 1
	#print "bf"
	f = open(file, "r")
	x = f.readlines()
	numvect = len(x)
	#print "nv = ", numvect
	DIST = [None] * numvect
	for i in range(numvect):
		DIST[i] = [0.0] * numvect
	max_dist = 0.0
	min_dist = 100000.0
	avg_dist = 0.0
	t = 0
	for i in range(numvect):
		#print "i = ", i
		xx = x[i].split(' ')
		for j in range(numvect):
			DIST[i][j] = float(xx[j])
			DIST[j][i] = DIST[i][j]
			avg_dist = avg_dist + DIST[i][j]
			t = t + 1
			if DIST[i][j] < min_dist and DIST[i][j] > 0.0:
				min_dist = DIST[i][j]
		mx_dist = max(DIST[i])
		if mx_dist > max_dist:
			max_dist = mx_dist

	avg_dist = avg_dist / t
	#print "max dist = ", max_dist
	#print "min dist = ", min_dist
	#print "avg dist = ", avg_dist
	return [DIST, avg_dist, max_dist, min_dist, numvect]

def	read_distance_matrix2(file):
	cont = 1
	#print "bf"
	f = open(file, "r")
	x = f.readlines()
	numvect = len(x)
	#print "nv = ", numvect
	DIST = [None] * numvect
	for i in range(numvect):
		DIST[i] = [0.0] * numvect
	max_dist = 0.0
	min_dist = 100000.0
	avg_dist = 0.0
	t = 0
	for i in range(numvect):
		#print "i = ", i
		xx = x[i].split(' ')
		for j in range(numvect):
			DIST[i][j] = float(xx[j])
			DIST[j][i] = DIST[i][j]
			avg_dist = avg_dist + DIST[i][j]
			t = t + 1
			#if DIST[i][j] < min_dist and DIST[i][j] > 0.0:
			#	min_dist = DIST[i][j]
		mx_dist = max(DIST[i])
		mn_dist = max(DIST[i])
		if mx_dist > max_dist:
			max_dist = mx_dist
		if mn_dist < min_dist:
			min_dist = mn_dist


	avg_dist = avg_dist / t
	#print "max dist = ", max_dist
	#print "min dist = ", min_dist
	#print "avg dist = ", avg_dist
	return [DIST, avg_dist, max_dist, min_dist, numvect]



def	create_random_vectors(numvect, dim, maxdist):
	D = []
	for i in range(numvect):
		d = []
		for j in range(dim):
			d.append(random.random()*maxdist)
		D.append(d)
	return D

def	discrete(PDs, mn, mx, ns):
	ln = len(PDs)
	discP = [0] * (ns + 1)
	for i in range(ln):
		v = (PDs[i] - mn) / (mx - mn)
		v = int(v * ns)
		discP[v] = discP[v] + 1
	
	return discP



def	obtain_radius(theta, max_dist, min_dist):
	R = []
	R.append(min_dist - 0.01)
	#R.append(min_dist - 1.0)
	#R.append(0.0)
	t = min_dist
	while t <= max_dist:
	#while t < max_dist:
		R.append(t)
		t = t + theta
	R.append(max_dist)
	return R

"""
nR: Number of radios
R: list of radios
d: The distance
"""
def	fit_radius(nR, R, d):
	L = []
	for i in range(nR - 1, -1, -1):
	#for i in range(nR - 1):
		if d <= R[i]:
		#if d < R[i]:
			L.append(i)
		else:
			return L
	return L


def	read_radius(file):
	f = open(file, "r")
	x = f.readlines()
	f.close()
	R = []
	for i in range(len(x)):
		R.append(float(x[i]))
	return R

def	entropy(L):
	H = 0.0
	for i in range(len(L)):
		p = L[i]
		if p > 0.0:
			h = p * (math.log(p) / math.log(2.0))
		else:
			h = 0.0
		H = H + h
	return -H

def	write_distance_matrix(FILE, D):
	f = open(FILE, "w")
	for i in range(len(D)):
		for j in range(len(D[i])):
			f.write(str(D[i][j]))
			f.write(" ")
		f.write("\n")
	f.close()

def	write_extras(FILE, r_N, r_l, v_r_N, last_lonely, avg_dist, min_dist, max_dist):
	f = open(FILE, "w")
	# The r for which at least one vector is neighbor of all others
	f.write(str(r_N))
	f.write("\n")
	# The r for which all vectors have at least one neighbor
	f.write(str(r_l))
	f.write("\n")
	# M: the list of vectors that have all the rest as neighbors.
	f.write("M\n")
	for i in range(len(v_r_N)):
		f.write(str(v_r_N[i]))
		f.write("\n")
	f.write("Last_lonely\n")
	for i in range(len(last_lonely)):
		f.write(str(last_lonely[i]))
		f.write("\n")
	f.write("---\n")
	f.write(str(avg_dist))
	f.write("\n")
	f.write(str(min_dist))
	f.write("\n")
	f.write(str(max_dist))
	f.write("\n")
	f.close()


def	save_distance_histogram(File_DH, PD_n, PD):
	# write the histogram of distances
	g = open(File_DH, "w")
	for i in range(len(PD_n)):
		g.write(str(i))
		g.write(" ")
		g.write(str(PD_n[i]))
		g.write("\n")
	g.close()

	# write the ratio of probabilities for consecutive distances
	g = open(sys.argv[11], "w")
	for i in range(len(PD)):
		g.write(str(i))
		g.write(" ")
		g.write(str(PD[i][0]))
		g.write(" ")
		g.write(str(PD[i][1]))
	g.write("\n")
	g.close()

def	write_info_R(FILE_R, nR, R, WhichNeigh, numvect):
	print "writing vectors"
	f = open(FILE_R, "w")
	for radius in range(nR):
		for i in range(len(WhichNeigh[radius])):
			f.write(str(radius) + "\t")
			f.write(str(i))
			f.write("\t")
			f.write(str(R[radius]) + "\t")
			f.write(str(float(float(len(WhichNeigh[radius][i]))/numvect)))
			f.write("\t")
			for nn in range(len(WhichNeigh[radius][i])):
				f.write(str(WhichNeigh[radius][i][nn]))
				f.write("\t")
			f.write("\n")
	f.close()

def	write_Etta(File_Etta, Etta, Kappa, nR, R):
	f = open(File_Etta, "w")
	for radius in range(nR):
		f.write(str(R[radius]))
		f.write(" ")
		f.write(str(Etta[radius]))
		f.write("\n")
	f.close()
	f = open(File_Etta + "_Kappa", "w")
	for radius in range(nR):
		f.write(str(R[radius]))
		f.write(" ")
		f.write(str(Kappa[radius]))
		f.write("\n")
	f.close()

def	write_info_vect(FILE_info, numvect, nR, R, InfoNN):
	print "writing vectors info"

	f = open(FILE_info, "w")
	for i in range(numvect):
		for radius in range(nR):
			f.write(str(i) + "\t")
			f.write(str(radius))
			f.write("\t")
			f.write(str(R[radius]))
			f.write("\t")
			ln = len(InfoNN[i][radius])
			f.write(str(ln) + "\t")
			for j in range(ln):
				f.write(str(InfoNN[i][radius][j]))
				f.write("\t")
			f.write("\n")
	f.close()

def	write_pdnn(numvect, nR, R, NNeigh, FILE_PDNN, type):
	inc = 1.0 / float(numvect)
	#inc = 1.0 / float(numvect - 1)
	inc2 = inc / 8.0
#print "inc = ", inc, " numvect = ", numvect
	f = open(FILE_PDNN, "w")
	for i in range(nR):
		nrn = 0.0
		for j in range(numvect):
			f.write(str(i))
			f.write("\t")
			f.write(str(R[i]))
			f.write("\t")
			f.write(str(nrn))
			f.write("\t")
			f.write(str(NNeigh[i][j]))
			f.write("\n")
			nrn = nrn + inc
		f.write("\n")
		"""
		nrn = 0.0
		for j in range(numvect):
			f.write(str(i))
			f.write(" ")
			f.write(str(R[i]+0.00001))
			f.write(" ")
			f.write(str(nrn))
			f.write(" ")
			f.write("0.0")
			f.write("\n")
			nrn = nrn + inc
		f.write("\n")
		"""
	f.close()

def	save_stats(FILE_stats, nR, numvect, NNeigh, R, H, E):
	f = open(FILE_stats, "w")
	for i in range(nR):
		mmx = -10.0
		mmn = 10000.0
		#for j in range(mv[i]):
		for j in range(numvect):
			if NNeigh[i][j] > mmx:
				mmx = NNeigh[i][j]
				ww = j
			if NNeigh[i][j] < mmn and NNeigh[i][j] > 0.0:
				mmn = NNeigh[i][j]
				xx = j
		# The radius r
		f.write(str(R[i]))
		f.write("\t")
		# M (the most probable number of neighbors)
		f.write(str(float(ww) / numvect))
		f.write("\t")
		# The largest probability
		f.write(str(mmx))
		f.write("\t")
		# m (the less probable number of neighbors, prob > 0)
		f.write(str(float(xx) / numvect))
		f.write("\t")
		# The smallest probability, prob > 0
		f.write(str(mmn))
		f.write("\t")
		# Entropy
		f.write(str(H[i]))
		f.write("\t")
		# The expected NN (number of neighbors)
		f.write(str(E[i]))
		f.write("\n")
	f.close()

"""
rad: the strategy to select radios
max_rad: diameter of the space
min_dist: minimum distance
theta: the number of radios to be selected
PD: the distance histogram, only to be used for strategy 1
nR: number of radii
Return:
[list of radios]
"""
def	select_radii(rad, max_dist, min_dist, theta, PD, nR, R_File):
	if rad == 0:
		inc = (max_dist - min_dist) / theta
		#theta = (avg_dist - min_dist) / theta
		R = obtain_radius(inc, max_dist, min_dist)
	else:
		if rad == 1:
			# select ten radius equally distributed
			N = 10
			theta = (max_dist - min_dist) / N
			Rs = obtain_radius(th, max_dist, min_dist)
			# the remaining radius are selected based on the ratios of
			# pf probabilities of distance
			Rt = PD[0:int(theta)-N]
			Ru = []
			for i in range(len(Rt)):
				Ru.append(Rt[i][1]-1)
				Ru.append(Rt[i][1])
			Rx = []
			for i in range(len(Rs)):
				Rx.append(Rs[i])
			for i in range(len(Ru)):
				Rx.append(Ru[i])
			print "Rs = ", Rx
			R = sorted(Rx)
		else:
			if rad == 2:
				R = read_radius(R_File)

	return R


"""
numvect: number of vectors
nR: number of radios
DIST: DIstance matrix

Returns:
[InfoNN, WhichNeigh, NNeigh] :

InfoNN is a matrix of size numvect x nR. Each element (i, r) of InfoNN
contains the list of neighbours of i when considering a radius r.

WhichNeigh is a matrix of size nR x numvect. Each element (r, i) of it
contains the list of vectors that has i neighbors at radius r.


NNeigh is a matrix of size nR x numvect. Each element (r,i) of NNeigh
contains the number of neighbours (probability, relative frequency) of vector i
when considering a radius r.
"""
def	obtain_Gamma(numvect, nR, DIST, R):
	NNeigh = [None] * nR
	for i in range(nR):
		NNeigh[i] = [0.0] * (numvect + 1)

	# Lets identify what vectors have that number of neighbors
	WhichNeigh = [None] * nR
	for i in range(nR):
		WhichNeigh[i] = [None] * numvect
		for j in range(numvect):
			WhichNeigh[i][j] = []

	InfoNN = [None] * numvect
	for i in range(numvect):
		InfoNN[i] = [None] * nR
		for j in range(nR):
			InfoNN[i][j] = []

	mv = [0] * nR
	for i in range(numvect):
		if i > 0 and i % 100 == 0:
			print "analyzing vectors", i - 100, "to", i
		nv = [0] * nR
		for j in range(numvect):
			if i != j:
				L = fit_radius(nR, R, DIST[i][j])
				for k in L:
					nv[k] = nv[k] + 1
					if nv[k] > mv[k]:
						mv[k] = nv[k]
					InfoNN[i][k].append(j)
		for k in range(nR):
			s = nv[k]
			NNeigh[k][s] = NNeigh[k][s] + 1
			WhichNeigh[k][s].append(i)
	for i in range(nR):
		for j in range(numvect):
		#for j in range(mv[i]):
			NNeigh[i][j] = NNeigh[i][j] / numvect

	return [InfoNN, WhichNeigh, NNeigh]

"""
nR: number of radios
R: list of radios
...
Return [Etta, Kappa]

Etta is the average absolute difference of the number of neighbours over all
pairs of vectors
"""
def	compute_Etta(nR, R, InfoNN, NNeigh, numvect):
	print "creating Etta"
	Etta = [0.0] * nR
	Kappa = [0.0] * nR
	for radius in range(nR):
		print "radius = ", R[radius]
		for i in range(numvect):
			#print "i = ", i
			for j in range(numvect):
				#if j % 100 == 0:
				#	print "j = ", j
				if i != j:
					Etta[radius] = Etta[radius] + abs(float(len(InfoNN[i][radius])) - float(len(InfoNN[j][radius]) ) )
					Kappa[radius] = Kappa[radius] + abs(NNeigh[radius][i] - NNeigh[radius][j] )
		Etta[radius] = Etta[radius] / float(numvect * numvect)
		Kappa[radius] = Kappa[radius] / float(numvect)

	return [Etta, Kappa]


"""
return [entropy of each PDF_r]
"""
def	obtain_entropy(nR, NNeigh):
	H = []
	for r in range(nR):
		h = entropy(NNeigh[r])
		H.append(h)

	return H

def	obtain_extras(WhichNeigh, numvect, R):
	w = -1
	for i in range(len(WhichNeigh)):
		#print WhichNeigh[i]
		if len(WhichNeigh[i][numvect-1]) > 0: 
			r_N = R[i]
			w = i
			break

	#print "r_N = ", r_N, w, WhichNeigh[w]
	if w == -1:
		v_r_N = [-1]
		r_N = -1
	else:
		v_r_N = WhichNeigh[w][numvect-1]
	#print "v(r_N) = ", v_r_N
	q = -1
	for i in range(len(WhichNeigh)):
		if len(WhichNeigh[i][0]) == 0:
			r_l = R[i]
			q = i
			break

	if q == -1:
		last_lonely = [-1]
		r_l = -1
	else:
		last_lonely = WhichNeigh[q-1][0]
		if q > 0:
			r_l = R[q-1]
		#last_lonely = WhichNeigh[q-1][0]

	return [r_N, r_l, v_r_N, last_lonely]

def	compute_divergence_PDF(NNeigh, nR, R):
	dPDF = []
	for i in range(1, nR):
		D = jensen_shannon(NNeigh[i], NNeigh[i-1])
		dPDF.append([R[i], D])
	return dPDF

def	compute_wasserstein_PDF(NNeigh, nR, R):
# from:
# https://github.com/wmayner/pyemd
	dPDF = []
	Ar = len(NNeigh[0])
	KF = 1.0/float(Ar)
	"""
	dmx = []
	for i in range(Ar):
		tmp = [0.0] * Ar
		tmp[i] = KF
		dmx.append(tmp)
	"""
	dmx = [None] * Ar
	for i in range(Ar):
		dmx[i] = [0.0] * Ar
	for i in range(Ar):
		for j in range(i+1, Ar):
			dmx[i][j] = float(j - i) * KF
			dmx[j][i] = dmx[i][j]
	print "dmx = ", len(dmx), len(dmx[0])
	dmx = numpy.array(dmx)
	V = []
	for i in range(nR):
		tmp = numpy.array(NNeigh[i], dtype=numpy.float)
		V.append(tmp)
	print "V = ", len(V), len(V[0])
	for i in range(1, nR):
		print "i = ", i
		#P = numpy.array(NNeigh[i], dtype=numpy.float)
		#Q = numpy.array(NNeigh[i-1], dtype=numpy.float)
		#print "P = ", P, len(P)
		#print "Q = ", Q, len(Q)
		#D = emd(P, Q, dmx)
		D = emd(V[i], V[i-1], dmx)
		dPDF.append([R[i], D])
	return dPDF


def	compute_KS(NNeigh, nR, R):
	KS_L = []
	for i in range(1, nR):
		ks = stats.ks_2samp(NNeigh[i], NNeigh[i-1])
		KS_L.append([R[i], ks[1]])
	return KS_L

def	write_divergence_PDF(FF, dPDF):
	f = open(FF, "w")
	for i in range(len(dPDF)):
		f.write(str(dPDF[i][0]) + '\t' + str(dPDF[i][1]) + '\n')
	f.close()

def	expected_NN(nR, R, numvect, NNeigh):
	inc = 1.0/float(numvect)
	E = []
	for r in range(nR):
		nrn = 0.0
		e = 0.0
		for j in range(numvect):
			e = e + nrn * NNeigh[r][j]
			nrn = nrn + inc
		E.append(e)
	return E

"""
python edapfun.py -d ../data/volcanoes/volcanoes.csv -save_DH -metric fd -r 30 -nr 0 -Etta -save_vec -save_R -o ../data/volcanoes/volcanoes_fd

This program obtains the probability distribution for the number of neighbors. A neighbor is a vector located
at a distance of theta or less
It is a generalization of pdnngh, but here, the range is defined automatically in terms of distance
If the last argument is 1, then it is read the data position, else, it is read the distance matrix
"""

"""
X = [0.0, 0.0, 0.1, 0.2, 0.2, 0.9, 0.1, 0.17]
#Y = [0.0, 0.0, 0.1, 0.2, 0.2, 0.9, 0.1, 0.17]
Y = [0.1, 0.0, 0.1, 0.2, 0.2, 0.6, 0.1, 0.15]

D1 = jensen_shannon(X, Y)
D2 = jensen_shannon(Y, X)
print "X = ", X
print "Y = ", Y
print "D = ", D1, D2
cc = sys.stdin.read(1)
"""

parser = argparse.ArgumentParser()
parser.add_argument('-d', action="store", dest="d", help="Input data")

parser.add_argument("-save_DM", action="store_true", dest="sDM", help="Save computed distance matrix")

parser.add_argument("-save_DH", action="store_true", dest="sDH", help="Save distance histogram")

parser.add_argument("-RFile", help="The file containing the radii for neighbourhood computation")

parser.add_argument('-metric', action="store", dest="metric", help="Metric")

parser.add_argument('-r', action="store", dest="num_rad", help="Number of radiuses", type=int)

parser.add_argument('-nr', action="store", dest="rad", help="Radiuses selection scheme", type=int)

parser.add_argument("-save_vec", action="store_true", dest="save_vectors", help="Save vectors info")

parser.add_argument("-save_R", action="store_true", dest="save_R", help="Save Info")

parser.add_argument("-dPDF", action="store_true", dest="dPDF", help="Compute divergence between consecutive PDFs")

parser.add_argument("-KS", action="store_true", dest="KS", help="Compute Kolmogorov-Smirnof between consecutive PDFs")

parser.add_argument("-Etta", action="store_true", dest="Etta", help="Compute Etta")

parser.add_argument('-o', action="store", dest="FILE", help="Output file")

args = parser.parse_args()

Input = args.d
print "Input = ", Input

# Data is always data vectors, not distance matrix
D_dM = 1

#if D_dM == 1:
[Dats, numvect, dim] = read_data(args.d)
# How to select radii
rad = int(args.rad)
# The number of radii
theta = args.num_rad
# Select radiuses...

try:
	if len(args.RFile) > 2 :
		R_File = args.RFile
	else:
		R_File = ""
except:
	R_File = ""

print "Computing distance matrix under metric ", args.metric
[DIST, avg_dist, max_dist, min_dist, PD, PD_n] = compute_distance_matrix(numvect, Dats, args.metric, rad, theta)

R = select_radii(rad, max_dist, min_dist, theta, PD, theta, R_File)
nR = len(R)
# R is the list of radius

print "The radius R = ", R, numvect

# DIST is the distance matrix for the numvect vectors
# PD the radiuses with highest ratio in PDFr / PDFr-1

FILE = args.FILE
if args.sDM == True:
	write_distance_matrix(FILE+'.DM', DIST)
print "...Done!"
#else: (if D_dM != 1)
	# Not supported for now
#	E = read_distance_matrix2(sys.argv[1])
#	numvect = E[4]

#write_distance_matrix(sys.argv[7], DIST)

if args.sDH == True:
	save_distance_histogram(FILE+'.DHist', PD_n, PD)

#PD = PD[0:num_rad]


"""
NNeigh is a matrix of size nR x numvect. Each element (r,i) of NNeigh
contains the number of neighbours (probability, relative frequency) of vector i
when considering a radius r
InfoNN is a matrix of size numvect x nR. Each element (i, r) of InfoNN
contains the list of neighbours of i when considering a radius r.
WhichNeigh is a matrix of size nR x numvect. Each element (r, i) of it
contains the list of vectors that has i neighbors at radius r.

"""
[InfoNN, WhichNeigh, NNeigh] = obtain_Gamma(numvect, nR, DIST, R)
if args.save_vectors == True:
	# Write the vectors from pd(R, nn)
	#write_info_vect(FILE + '_vect.csv', nR, R, WhichNeigh, numvect)
	write_info_vect(FILE + '_vect.csv', numvect, nR, R, InfoNN)

if args.save_R == True:
	write_info_R(FILE + '_R.csv', nR, R, WhichNeigh, numvect)
	#write_info_R(FILE + '_R.csv', numvect, nR, R, InfoNN)

if args.Etta == True:
	[Etta, Kappa] = compute_Etta(nR, R, InfoNN, NNeigh, numvect)
	write_Etta(FILE + '.Etta', Etta, Kappa, nR, R)

if args.dPDF == True:
	print "computing KL for PDFs"
	dPDF = compute_divergence_PDF(NNeigh, nR, R)
	#print "dPDF = ", dPDF
	write_divergence_PDF(FILE + '.dPDF', dPDF)
	# The Wasserstein distance
	dPDF = compute_wasserstein_PDF(NNeigh, nR, R)
	write_divergence_PDF(FILE + '.dPDF_W', dPDF)

if args.KS == True:
	print "computing Kolmogorov-Smirnoff over consecutive PDFs"
	KS = compute_KS(NNeigh, nR, R)
	write_divergence_PDF(FILE + '.KS', KS)

#type = 0 is the usual
#type = 1 is for more fancy visualization in gnuplot
type = 0
write_pdnn(numvect, nR, R, NNeigh, FILE+'.PDNN', type)

E = expected_NN(nR, R, numvect, NNeigh)

# Obtain the entropy of the PDF(r)
H = obtain_entropy(nR, NNeigh)
save_stats(FILE+'.stats', nR, numvect, NNeigh, R, H, E)

# Now, obtain r_N, r_l
[r_N, r_l, v_r_N, last_lonely] = obtain_extras(WhichNeigh, numvect, R)
write_extras(FILE+'.extras', r_N, r_l, v_r_N, last_lonely, avg_dist, min_dist, max_dist)

# **** ploT:
# splot for [i=0:68] 'vals_055_Normal.pdnn' every :::(2*i)::(2*i+1) u 2:3:4:1 w pm3d
# splot for [i=0:74] 'vals_055_Normal.pdnn' every :::(2*i)::(2*i+1) u 2:3:4:1 w pm3d, for [i=0:66] 'vals_055_Normal.pdnn' every :::(2*i)::(2*i+1) u 2:3:4 w l lt -1
