"""
    vec_var_index.py implements the idea of inspecting high-dimensional datasets
    by means of the probability density function of the number of neighbors.
    Copyright (C) 2018  Antonio Neme Castillo.

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


import	sys, argparse

def	is_in(a, L):
	for i in L:
		if a == i:
			return 1
	return -1

def	read_R(FF, NB, N):
	f = open(FF, "r")
	x = f.readlines()
	f.close()

	# The bucket for vector i at radius r
	Vec = {}
	# The list of vectors in bucket n at radius r
	Bucket = {}

	# The bucket n for vector i at radius r. This looks into the number of
	# vectors
	VecN = {}
	BucketN = {}

	Radii = []
	for i in x:
		xx = i.split('\t')
		r = float(xx[2])
		if is_in(r, Radii) == -1:
			Radii.append(r)
		NumNeigh = len(xx) - 4 - 1
		#NumNeigh = len(xx) - 4
		if NB == -1:
			buck = float(xx[3])
		else:
			buck = int(float(xx[3])*NB)
		# Here is the problem! (r, buck) should not be reset
		# Look for example, for circle3, lines 10074 and 10075
		# my_dict[some_key] = my_dict.get(some_key, 0) + 1
		# https://stackoverflow.com/questions/473099/check-if-a-given-key-already-exists-in-a-dictionary-and-increment-it
		#Bucket[r, buck] = []
		#Bucket[r, buck].append(Bucket.get((r,buck), []))
		if (r,buck) not in Bucket:
			Bucket[r, buck] = []

		buckN = float(xx[1])/N
		BucketN[r, buckN] = []
		for j in range(NumNeigh):
			v = int(xx[4 + j])
			Vec[v, r] = buck
			Bucket[r, buck].append(v)

			VecN[v, r] = buckN
			BucketN[r, buckN].append(v)

	return [Vec, Bucket, Radii, VecN, BucketN]

def	read_vec(FF, num):
	f = open(FF, "r")
	x = f.readlines()
	f.close()

	VecInfo = {}
	for i in x:
		xx = i.split('\t')
		r = float(xx[2])
		nn = float(xx[3])/num
		v = int(xx[0])
		VecInfo[v, r] = nn
	return VecInfo

def	save_index(FF, num, Radii, IdxAvgAbs, IdxAvg, IdxMaxAbs, IdxAvgAbsN, IdxAvgN, IdxMaxAbsN):
	f = open(FF, "w")
	for i in range(num):
		for r in range(len(Radii)-1):
			f.write(str(i) + '\t' + str(r) + '\t' + str(Radii[r]) + '\t' + str(IdxAvgAbs[(i, Radii[r])]) + '\t' + str(IdxAvg[(i, Radii[r])]) + '\t' + str(IdxMaxAbs[(i, Radii[r])]) + '\t' +   str(IdxAvgAbsN[(i, Radii[r])]) + '\t' + str(IdxAvgN[(i, Radii[r])]) + '\t' + str(IdxMaxAbsN[(i, Radii[r])]) +  '\n')
	f.close()

#def	save_space_nn_index(FF, num, Radii, IdxAvgAbs, IdxAvg, IdxMaxAbs, VecInfo):
def	save_space_nn_index(FF, num, Radii, IdxAvgAbs, IdxAvg, IdxMaxAbs,  IdxAvgAbsN, IdxAvgN, IdxMaxAbsN, VecInfo):
	ln = len(Radii) - 1
	f = open(FF, "w")
	f.write("#vec\tr\ts\tRad_r\tRad_s\tIdxAvgAbs\tIdxAvg\tIdxMaxAbs\tVecInfo\tIdxAvgAbsN\tIdxAvgN\tIdxMaxAbsN\n")
	for i in range(num):
		for r in range(ln):
			for s in range(ln):
			#for s in range(r+1, ln):
				f.write(str(i) + '\t' + str(r) + '\t' + str(s) + '\t' + str(Radii[r]) + '\t' + str(Radii[s]) + "\t" + str(IdxAvgAbs[(i, Radii[r], Radii[s])]) + '\t' + str(IdxAvg[(i, Radii[r], Radii[s])]) + '\t' + str(IdxMaxAbs[(i, Radii[r], Radii[s])]) + '\t' + str(VecInfo[(i, Radii[r])]) + '\t' + str(IdxAvgAbsN[(i, Radii[r], Radii[s])]) + '\t' + str(IdxAvgN[(i, Radii[r], Radii[s])]) + '\t' + str(IdxMaxAbsN[(i, Radii[r], Radii[s])]) + '\n')
	f.close()

def	save_space_nn_index_averages(FF, num, Radii, IdxAvgAbs, IdxAvg, IdxMaxAbs,  IdxAvgAbsN, IdxAvgN, IdxMaxAbsN, VecInfo):
	ln = len(Radii) - 1

	Avgs = {}
	N = {}
	for i in range(ln):
		for j in range(ln):
			for k in range(ln):
				Avgs[(i,j,k)] = 0.0
			N[(i,j)] = 0

	for i in range(num):
		for j in range(ln):
			RR = Radii[j]
			for k in range(j + 1, ln):
				SS = Radii[k]
				Avgs[(j,k,0)] = Avgs[(j,k,0)] + IdxAvgAbs[(i,RR,SS)]
				Avgs[(j,k,1)] = Avgs[(j,k,1)] + IdxAvg[(i,RR, SS)]
				Avgs[(j,k,2)] = Avgs[(j,k,2)] + IdxMaxAbs[(i,RR,SS)]
				Avgs[(j,k,3)] = Avgs[(j,k,3)] + VecInfo[(i,RR)]
				Avgs[(j,k,4)] = Avgs[(j,k,4)] + IdxAvgAbsN[(i,RR,SS)]
				Avgs[(j,k,5)] = Avgs[(j,k,5)] + IdxAvgN[(i,RR,SS)]
				Avgs[(j,k,6)] = Avgs[(j,k,6)] + IdxMaxAbsN[(i,RR,SS)]
				N[(j,k)] = N[(j,k)] + 1 

	f = open(FF, "w")
	for r in range(ln):
		for s in range(ln):
			cmd = ""
			for k in range(6):
				cmd = cmd + str(Avgs[(r, s, k)]) + "\t"
				#cmd = cmd + str(Avgs[(r, s, k)]/num) + "\t"
			cmd = cmd + str(Avgs[(r, s, 6)]/num)
			f.write(str(r) + "\t" + str(s) + "\t" + str(Radii[r]) + "\t" + str(Radii[s]) + "\t" + cmd + "\n")
		f.write("\n")
	f.close()




"""
python vec_var_index.py -R ../data/circle2clust/circle2clust_euclidean_R.csv -o ../data/circle2clust/circle2clust_euclidean_index.csv -num 1200

python vec_var_index.py -R ../data/iris/iris_euclidean_R.csv -o ../data/iris/iris_euclidean_index.csv -num 150 -s ../data/iris/iris_euclidean_nn_idx.csv -vec ../data/iris/iris_euclidean_vect.csv 

python vec_var_index.py -R /Volumes/vitD_etal/neme/edapfun/data/circle3/circle3_euclidean_R.csv -o ../data/circle3/circle3_euclidean_index.csv -num 5034 -s ../data/circle3/circle3_euclidean_nn_idx.csv -vec /Volumes/vitD_etal/neme/edapfun/data/circle3/circle3_euclidean_vect.csv

plot "circle3_euclidean_nn_idx.csv" using 5:($2 == 3? $6:1/0) ls 1

This program computes the "trajectory divergence" between a vector i and
the set of vectors V(B_r^i) that are at the same bucket for radius r at the
next radius s (s = r + increment for radii).

Let B_r^i be the bucket at which vector i is located for radius r.

Let V(B_r^i) be the list of vectors located in the same bucket as vector i
(all vector in this bucket have B_r^i neighbours).

Let B_s^i be the bucket for vector i at radius s

For each vector v in V(B_r^i), find B_s^v.
T_i = f(d(B_s^i, B_s^v))
"""

parser = argparse.ArgumentParser()
parser.add_argument('-R', action = "store", dest = "R", help = "The information aboout radii and vectors (_R.csv)")
parser.add_argument('-o', action = "store", dest = "o", help = "The output file")
#parser.add_argument('-o2', action = "store", dest = "o2", help = "The second output file")
parser.add_argument('-num', action = "store", dest = "num", help = "The number of vectors")
parser.add_argument('-NB', dest = "NB", help = "The number of bins for the histogram (optional)")
parser.add_argument('-vec', dest = "vec", help = "The information about vectors (_vect.csv), optional")
parser.add_argument('-s', dest = "s", help = "The output file for the (numneigh, index) space, optional")
#parser.add_argument('-s2', dest = "s2", help = "The output file for the (numneigh, index) space, optional, N")
#parser.add_argument('-dn', action = "store", dest = "dn", help = "The delta n, the buckets from n-dn to n+dn")

args = parser.parse_args()
if args.NB:
	NB = int(args.NB)
else:
	NB = -1

num = int(args.num)

[Vec, Buckets, Radii, VecN, BucketsN] = read_R(args.R, NB, num)

if args.vec:
	VecInfo = read_vec(args.vec, float(num))

#dn = int(args.dn)

# The index for each vector, as the average
IdxAvgAbs = {}
IdxAvg = {}
# The index for each vector, as the maximum difference...
IdxMaxAbs = {}

# For N
# The index for each vector, as the average
IdxAvgAbsN = {}
IdxAvgN = {}
# The index for each vector, as the maximum difference...
IdxMaxAbsN = {}

ln = len(Radii)-1
for i in range(num):
	for r in range(ln):
		# The bucket in which vector i is at radius r
		buck = Vec[(i, Radii[r])]
		# The similar vectors (Those in the same bucket)
		VB = Buckets[(Radii[r], buck)]
		# Consider N (number of neighbors)
		# The bucket in which vector i is at radius r
		buck = VecN[(i, Radii[r])]

		# The similar vectors (Those in the same bucket)
		VBN = BucketsN[(Radii[r], buck)]

		for s in range(ln):
		#for s in range(r+1, ln):
			# The bucket in which vector i is located at the next radius
			buck_s = Vec[(i, Radii[s])]
			# The vectors in the same bucket as vector i for radius r.
			# These are the vectors "similar" to vector i
			distAbs = 0.0
			dist = 0.0
			distMX = 0.0
			#print "i = ", i, Radii[r], buck, buck_s, VB
			for v in VB:
				# identify the bucket of each of the similar vectors
				buckv_s = Vec[(v, Radii[s])]
				vva = abs(buck_s - buckv_s)
				distAbs = distAbs + vva
				if vva > distMX:
					distMX = vva
				dist = dist + buck_s - buckv_s
			if len(VB) > 0:
				IdxAvgAbs[i, Radii[r], Radii[s]] = distAbs/float(len(VB))
				IdxAvg[i, Radii[r], Radii[s]] = dist/float(len(VB))
				IdxMaxAbs[i, Radii[r], Radii[s]] = distMX
			else:
				IdxAvgAbs[i, Radii[r], Radii[s]] = 0.0
				IdxAvg[i, Radii[r], Radii[s]] = 0.0
				IdxMaxAbs[i, Radii[r], Radii[s]] = 0.0

			# The bucket in which vector i is located at the next radius
			buck_s = VecN[(i, Radii[s])]
			# The vectors in the same bucket as vector i for radius r.
			# These are the vectors "similar" to vector i
			distAbs = 0.0
			dist = 0.0
			distMX = 0.0

			"""
			if r > 1 and s < r:
				print "V = ", VBN
				print "bk = ", buck
				print "bs = ", buck_s
				#cc = sys.stdin.read(1)
			"""
			#print "i = ", i, Radii[r], buck, buck_s, VB
			for v in VBN:
				# identify the bucket of each of the similar vectors
				buckv_s = VecN[(v, Radii[s])]
				vva = abs(buck_s - buckv_s)
				distAbs = distAbs + vva
				if vva > distMX:
					distMX = vva
				dist = dist + buck_s - buckv_s
			if len(VBN) > 0:
				IdxAvgAbsN[i, Radii[r], Radii[s]] = distAbs/float(len(VBN))
				IdxAvgN[i, Radii[r], Radii[s]] = dist/float(len(VBN))
				IdxMaxAbsN[i, Radii[r], Radii[s]] = distMX
			else:
				IdxAvgAbsN[i, Radii[r], Radii[s]] = 0.0
				IdxAvgN[i, Radii[r], Radii[s]] = 0.0
				IdxMaxAbsN[i, Radii[r], Radii[s]] = 0.0
			"""
			if r > 1 and s < r and s == 1:
				print "i = ", i, r, buck, buck_s, buckv_s, IdxMaxAbsN[i, Radii[r], Radii[s]]
				print "dist = ", distAbs, dist, distMX
				cc = sys.stdin.read(1)
			"""

	if i % 100 == 0:
		print "Processing vector ", i

#save_index(args.o, num, Radii, IdxAvgAbs, IdxAvg, IdxMaxAbs, IdxAvgAbsN, IdxAvgN, IdxMaxAbsN)

if args.s and args.vec:
	save_space_nn_index(args.s, num, Radii, IdxAvgAbs, IdxAvg, IdxMaxAbs, IdxAvgAbsN, IdxAvgN, IdxMaxAbsN, VecInfo)
	save_space_nn_index_averages(args.o, num, Radii, IdxAvgAbs, IdxAvg, IdxMaxAbs, IdxAvgAbsN, IdxAvgN, IdxMaxAbsN, VecInfo)
