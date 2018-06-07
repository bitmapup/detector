##!/usr/bin/env python
"""
Provides the methods for calculating four distances measures: Principal Component Analysis (PCA), Euclidean distance, Distance Time Warping (DTW) and Least Common Subsequence String (LCSS). Also, has the methods to compute the Normal Discounted Cumulative Gain (NDCG) of each ranking and so determine which distance measure and which trajectory should be selected.
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA as sklearnPCA
import sys,os
import mlpy
from math import pow
from math import sqrt
from math import sin
from math import cos
from math import atan
str_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str_path)
from query import *
import datetime
import math
import collections
import logging
####################################################################
# To represent trajectories we use numpy matrices pf shape 3 x n   #
# where first row is the timestamp, the second is latitude and the #
# last one is longitude                                            #
####################################################################

def test_pca_distance():
    #Getting data
    for i in os.listdir("sample"):
    	U1 = "sample/"+i
    	for j in os.listdir("sample"):
    #U2 = "data/user_1684694296484880_1418.csv"
    #U1 = "data/user_1465644436375197_1418.csv"
    		U2 = "sample/"+j
    		antenne = getAntennasLocation ()
    		class1_sample = getMobilityTracesMatrix(U1,antenne)
    		class2_sample = getMobilityTracesMatrix(U2,antenne)
    		minutes=30
    		(u1,u2) = alignInTime(class1_sample,class2_sample,minutes)
    
    		pcaDist = pca_distance(u2,u1,False)
    		print "pca_distance: {0}".format(pcaDist)+","+"{0}".format(U1)+","+"{0}".format(U2)

#end test_pca_distanca

def test_pca_distance_iteratively(path,path2):
    """
    Compute PCA distance between a trajectory and a set of trajectories.
    The trajectories ared ordered by distance
    @path: path of a trajectory
    @path2: path of a set of a trajectory
    """
    U1 = path
    resultados=dict()
    for j in os.listdir(path2):
	U2 ="{0}/".format(path2)+ j
	if U1!=U2:
		antenne = getAntennasLocation()
    		class1_sample = getMobilityTracesMatrix(U1,antenne)
    		class2_sample = getMobilityTracesMatrix(U2,antenne)
    		minutes=30
    		(u1,u2) = alignInTime(class1_sample,class2_sample,minutes)
    		pcaDist  = pca_distance(u1,u2,False)
		resultados[pcaDist]=U2
    ord_resultados=collections.OrderedDict(sorted(resultados.items(),reverse=True))
   
    tray=list()
    for key,value in ord_resultados.iteritems():
	tray.append(value)
    return tray


def pca_distance(class1_sample,class2_sample,plot=False):
    """
    Instead of working in trajectory coordinate space, PCA is used to transform 
    trajectory into a lower dimension subspace. Trajectories must have the same length
    PCA makes a dimension reduction from 3d (timestamp,lat,lon) to 2d (var_x, var_y)
    in order to perform the similarity calculus.
    @class1_sample is a numpy narray of 3xn
    @class2_sample is a numpy narray of 3xn
    @plot depicts the trajectories in 3d and 2d
    """
    if plot:
        #PLOT
        #%pylab inline
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10
        ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:],
                    'o', markersize=8, color='blue', alpha=0.5, label='trajectory1')
        ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:],
                        '^', markersize=8, alpha=0.5, color='red', label='trajectory2')

        plt.title('Samples for class 1 and class 2')
        ax.legend(loc='upper right')

        plt.show()

    #Mergin vectors
    all_samples = np.hstack((class1_sample, class2_sample))
    sklearn_pca = sklearnPCA(n_components=2)

    #print "shape1: {0}".format(class1_sample.shape[1])
    sklearn_transf = sklearn_pca.fit_transform(all_samples.T)


    if plot:
        plt.plot(sklearn_transf[0:class1_sample.shape[1],0],
                sklearn_transf[0:class1_sample.shape[1],1],
                         'o', markersize=7, color='blue', alpha=0.5, label='trajectory1')
        plt.plot(sklearn_transf[class1_sample.shape[1]:sklearn_transf.shape[0],0], 
                 sklearn_transf[class1_sample.shape[1]:sklearn_transf.shape[0],1],
                         '^', markersize=7, color='red', alpha=0.5, label='trajectory2')
        #plt.xlim([vmin[0],vmax[0]])
        #plt.ylim([vmin[1],vmax[1]])
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.legend()
        plt.title('Transformed samples using sklearn.decomposition.PCA()')
        plt.show()
    #vmax = numpy.amax(sklearn_transf,axis=0)
    #vmin = numpy.amin(sklearn_transf,axis=0)
    #print "min: {0} max {1}".format(vmin,vmax)

    #Computing distance
    class1_tsample = sklearn_transf[0:class1_sample.shape[1],]
    class2_tsample = sklearn_transf[class1_sample.shape[1]:sklearn_transf.shape[0],]
    #print "tshape: {0} - {1}".format(class1_tsample.shape,class2_tsample.shape)
    sum = 0
    i = 0
    size = class1_tsample.shape[0]
    while i < size:
        sum += euclidean_distance(class1_tsample[i],class2_tsample[i])
        i += 1

    return float(sum)/float(size)

#end pca_distance

def euclidean_distance(x,y):
    """
    Computes the Euclidean distance of two vectors of form [timestamp,lat,lon]
    @x is a a vector of form [timestamp,lat,lon], [lat,lon] or [x,y]
    @y is a a vector of form [timestamp,lat,lon], [lat,lon] or [x,y]
    """
    i=0
    if len(x)==len(y):
        sum = 0
        while i<len(x):
            sum += pow((float(x[i])-float(y[i])),2)
            i+=1
	    #print x,y
        return sqrt(sum)
    else:
        return -1
     
#def euclidean_distance

def distance_latlon(x,y):
    """
    Call the function to ompute the Earth mover distance of two vectors of form [timestamp,lat,lon]
    @x is a a vector of form [timestamp,lat,lon]
    @y is a a vector of form [timestamp,lat,lon]
    """
    return distance_only_latlon(x[1],x[2],y[1],y[2])

def distance_only_latlon(latitude1, longitude1,latitude2, longitude2):
    """
    Computes the Earth mover distance of two vectors of form [timestamp,lat,lon]
    @latitude1 latitude of the first cordinate
    @longitude1 longitude of the first cordinate
    @latitude2 latitude of the second cordinate
    @longitude2 longitude of the second cordinate
    """

    PIov180 = 0.017453292519943295
    dLat = (float(latitude2) - float(latitude1)) * PIov180
    dLon = (float(longitude2) - float(longitude1)) * PIov180
    a = sin(dLat/2)**2+(sin(dLon/2)**2)*cos(float(latitude1)*PIov180)*cos(float(latitude2)*PIov180)
    divisor = sqrt(1-a)

    if (divisor == 0):
         divisor = 1
    return round(12742 * atan(sqrt(a)/ divisor), 6)
#def distance_only_latlon

def test_linear_distance_iteratively(path,path2):
    """
    Computes the euclidean distances between a trajectory
    and a set of trajectories. Trajectories are ordered by distance
    @path: path of a trajectory
    @path2: path of a set of trajectories
    """
    #Getting data

    U1 = path
    resultados=dict()
    for j in os.listdir(path2):
	U2 ="{0}/".format(path2)+ j
	if U1!=U2:
		antenne = getAntennasLocation()
    		class1_sample = getMobilityTracesMatrix(U1,antenne)
    		class2_sample = getMobilityTracesMatrix(U2,antenne)
    		minutes=30
    		(u1,u2) = alignInTime(class1_sample,class2_sample,minutes)

    		distance  = linear_distance(u1,u2,earthMoverDistance=True)
		resultados[distance]=U2
    ord_resultados=collections.OrderedDict(sorted(resultados.items(),reverse=True))

    tray=list()
    for key,value in ord_resultados.iteritems():
	tray.append(value)
    return tray
    		


def test_linear_distance():
    #Getting data
	
    U1 = "data/user_1000408639311334.csv"
    U2 = "data/user_1000408639311334.csv"
    antenne = getAntennasLocation()
    class1_sample = getMobilityTracesMatrix(U1,antenne)
    #class2_sample = getMobilityTracesMatrix(U2,antenne)
    class2_sample = getMobilityTracesDistanceModified(U2,antenne,10)
    #print class1_sample,class2_sample
    minutes=30
    (u1,u2) = alignInTime(class1_sample,class2_sample,minutes)
 
   
    distance  = linear_distance(u1,u2,earthMoverDistance=True)
    #distance  = linear_distance(u1,u2,earthMoverDistance=False)
    print "linear: {0}".format(distance)+","+"{0}".format(U1)+","+"{0}".format(U2)
    

#def test_linear_distance

def linear_distance(class1_sample,class2_sample,earthMoverDistance=True):
    """
    Computes the linear distance between  two trajectories with the same length
    (class1_sample,class2_sample)
    based on Euclidean or Earth Mover Distance  disimilarity function
    @class1_sample: numpy matrix representing a trajectory
    @class2_sample: numpy matrix representing a trajectory
    @earthMoverDistance: boolean flag to use Earth Mover Distance (True) or 
    Euclidean (False)  disimilarity function 
    """
    #Computing distance
    sum = 0
    i = 0
    size = class1_sample.shape[1]
    while i < size:
        if earthMoverDistance:
           sum += distance_only_latlon( class1_sample[1,i],class1_sample[2,i],
                                        class2_sample[1,i],class2_sample[2,i])
        else:
           sum += euclidean_distance(class1_sample[:,i],class2_sample[:,i])
	   #print class1_sample[:,i]
   	   #print sum    
    	i += 1
    #print size
    #print sum
    #print class1_sample
    #print class2_sample

    return float(sum)/float(size)

#def linear_distancea


######################################################################
# TIME FILTERS
#####################################################################

def test_alignInTime():
    U1 = "data/user_1000408639311334.csv"
    U2 = "data/user_1004631430542939.csv"
    antenna = getAntennasLocation ()
    minutes=30
    (u1,u2) = alignInTime(U1,U2,antenna,minutes)
    print u1.shape
    print u2.shape


def areTimeOverlaping(class1_sample,class2_sample):
    """ 
      Verifies if both trajectories are time overlaping, and thus comparable.
      @class1_sample: user1 matrix representinx a trajectory in the form of 3xn
      @class2_sample: user2 matrix representinx a trajectory in the form of 3xn
    """
    class1_sample=class1_sample.T
    class2_sample=class2_sample.T

    t = False

    ymax1 = np.amax(class1_sample,axis=1)[0]
    ymax2 = np.amax(class2_sample,axis=1)[0]
    ymin1 = np.amin(class1_sample,axis=1)[0]
    ymin2 = np.amin(class2_sample,axis=1)[0]
    end = min(ymax1,ymax2)
    begin = max(ymin1,ymin2)
    interval = end - begin

    
    if interval>=0:
        t = True

    return t
#end areTimeOverlaping

def alignInTime(class1_sample,class2_sample,timeWindows):
    """
    return a matrix aligned in time by discretizing the time in timeWindows intervals:
    TIME : t0 -> t0+tw -> t0+2tw -> ... -> t0+ntw
    VALUE: x0 ->   x1  ->   x2   -> ... -> xn
    @class1_sample: user1 matrix representinx a trajectory in the form of nx3
    @class2_sample: user2 matrix representinx a trajectory in the form of nx3
    @timeWindows (tw): minutes to make the intervals
    """
    myformat='%Y-%m-%d %H:%M:%S'
    #ALl the process after takes a matrix of shape 3xn
    class1_sample=class1_sample.T
    class2_sample=class2_sample.T
    ymax1 = np.amax(class1_sample,axis=1)[0]
    ymax2 = np.amax(class2_sample,axis=1)[0]
    ymin1 = np.amin(class1_sample,axis=1)[0]
    ymin2 = np.amin(class2_sample,axis=1)[0]
    end = min(ymax1,ymax2)
    begin = max(ymin1,ymin2)

    interval = end - begin
    nb_intervals = round(interval/(int(timeWindows)*60))
    #print nb_intervals
    size1 = class1_sample.shape[1]
    size2 = class2_sample.shape[1]
    #print size1,size2
    list_class1 = list()
    list_class2 = list()
    
    if interval<0:
        raise ValueError('A very specific bad thing happened')
        return  -1

    list_timestamps = list()
    i = 0

    if (nb_intervals<1):
        nb_intervals = 1

    while i < nb_intervals:
       ts = begin + (i*int(timeWindows)*60)
       i += 1
       list_timestamps.append(int(ts))
    list_timestamps.append(int(end))
    #print list_timestamps

    #for l in list_timestamps:
    #    print datetime.datetime.fromtimestamp(l).strftime(myformat)

    
    dict_class1 = _processAlignInTime(class1_sample,list_timestamps)
    #print dict_class1
    dict_class2 = _processAlignInTime(class2_sample,list_timestamps)
    
    #cleaning empty time slots
    new_list_timestamps = list()
    for ts in list_timestamps:
        if ((ts in dict_class1) & (ts in dict_class2)):
            if ( (dict_class1[ts][0] == 0) | (dict_class2[ts][0] == 0) ): 
                del dict_class1[ts]
                del dict_class2[ts]
            else:
                list_class1.append(dict_class1[ts])
                list_class2.append(dict_class2[ts])
                new_list_timestamps.append(ts)

    return (np.array(list_class1).T, np.array(list_class2).T)
# alignInTime():

def _processAlignInTime(class_sample,list_timestamps):
    j = -1
    list_class =  [None] * (len(list_timestamps)-1)
    size = class_sample.shape[1]-1
    aux = list()
    myformat='%Y-%m-%d %H:%M:%S'

    dict_points={}

    for lts in list_timestamps:
        dict_points[lts]=[]

    while j < size:
     j += 1
     #First row is timestamp
     t = class_sample[0,j]
     i = 0
     while i<(len(list_timestamps)-2):
        ts_begin = list_timestamps[i]
        ts_end = list_timestamps[i+1] 
        #verifies if t is between ts_begin and ts_end
        if (ts_begin <= t)&(t < ts_end):
            dict_points[ts_begin].append(class_sample[:,[j]])
            i = i + 1
            break
        i = i + 1

    for key in dict_points:
        aux = dict_points[key]
        aux_size = len(aux)
        if aux_size==0: aux_size=1
        ts = 0
        lat = 0
        lon = 0

        for a in aux:
            ts += a[0][0]
            lat += a[1][0]
            lon += a[2][0]

        ts /= aux_size
        lat /= aux_size
        lon /= aux_size
        dict_points[key] = [int(ts),lat,lon]

    return dict_points
#def _processAlignInTime

def filterTime(class1_sample,class2_sample):
    """
    This procedure filter all point out of a timewindows.
    The timewindows is computed taking the maximal timestamp
    of the minimal timestams values of both trajectories and the 
    minimal timestamp of the maximal timestams values of both trajectories.
    Thus, at the end we have two trajectories at the same period.
    @class1_sample: user1 matrix representinx a trajectory 
    @class2_sample: user2 matrix representinx a trajectory 
    """

    vmin = max(numpy.amin(class1_sample,axis=0)[0],numpy.amin(class2_sample,axis=0)[0])
    vmax = min(numpy.amax(class1_sample,axis=0)[0],numpy.amax(class2_sample,axis=0)[0])

    i = len(class1_sample)-1
    j = len(class2_sample)-1
    
    print class1_sample.shape,class2_sample.shape

    while 0 <= i:
        print "{0}<= {1}<={2} {3} {4}".format(vmin,class1_sample[i,0],vmax,
                (vmin<=class1_sample[i,0]),(class1_sample[i,0] <= vmax))
        if not((vmin<=class1_sample[i,0]) and  (class1_sample[i,0] <= vmax)):
           np.delete(class1_sample, i, 0)
        i-=1
    while 0 <= j:
        print "{0}<= {1}<={2} {3} {4}".format(vmin,class2_sample[j,0],vmax,
                (vmin<=class2_sample[j,0]),(class2_sample[j,0] <= vmax))
        if not((vmin<=class2_sample[j,0]) and  (class2_sample[j,0] <= vmax)):
           np.delete(class2_sample, j, 0)
        j-=1

    print class1_sample.shape,class2_sample.shape
#end filterTime

######################################################################
# END TIME FILTERS
#####################################################################


def dwt(class1_sample,class2_sample,plot=False,dist=lambda x, y: np.linalg.norm(x - y, ord=1)):
    """
    dwt (Dynamic Time Wraping) function computes the distance between unequal trajectories.
    It finds  a time warping that minimize the total distance between matching points
    @class1_sample: user1 matrix representinx a trajectory 
    @class2_sample: user2 matrix representinx a trajectory 
    """
    x=class1_sample
    y=class2_sample

    r, c = len(x), len(y)

    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    for i in range(r):
        for j in range(c):
            D[i+1, j+1] = dist(x[i], y[j])

    for i in range(r):
        for j in range(c):
            D[i+1, j+1] += min(D[i, j], D[i, j+1],
            D[i+1, j])
    
    D = D[1:, 1:]
    dist = D[-1, -1] / sum(D.shape)
    cost = D
    path = _trackeback(D)

    if plot:
        plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.xlim((-0.5, cost.shape[0]-0.5))
        plt.ylim((-0.5, cost.shape[1]-0.5))
        plt.show()

    return dist, cost, path

#end dwt()
def _trackeback(D):
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while (i > 0 and j > 0):
        tb = np.argmin((D[i-1, j-1], D[i-1, j], D[i, j-1]))

        if (tb == 0):
            i = i - 1
            j = j - 1
        elif (tb == 1):
            i = i - 1
        elif (tb == 2):
            j = j - 1
        
        p.insert(0,i)
        q.insert(0,j)
    p.insert(0, 0)
    q.insert(0, 0)
    return (np.array(p), np.array(q))


def test_dtw_distance_iteratively(path,path2):
    """
    Computes DTW distance between a trajectory and a set of trajectories.
    The trajectories ared ordered by distance
    @path: path of a trajectory
    @path2: path of a set of trajectories
    """
    U1 = path
    resultados=dict()
    for j in os.listdir(path2):
        U2 ="{0}/".format(path2)+ j
        if U1!=U2:
                antenne = getAntennasLocation()
                class1_sample = getMobilityTracesMatrix(U1,antenne)
                class2_sample = getMobilityTracesMatrix(U2,antenne)
                dist, cost, path  = dwt(class1_sample,class2_sample,False)
                resultados[dist]=U2
    ord_resultados=collections.OrderedDict(sorted(resultados.items(),reverse=True))

    tray=list()
    for key,value in ord_resultados.iteritems():
        tray.append(value)
    return tray


def test_dtw_distance():
    #Getting data
    for i in os.listdir("data"):
	U1 = "data/"+i
    	for j in os.listdir("data"):
		U2 = "data/"+j
    #U1 = "data/user_1000408639311334.csv"
    #U2 = "data/user_1004631430542939.csv"
    		antenne = getAntennasLocation ()
    		class1_sample = getMobilityTracesMatrix(U1,antenne)
    		class2_sample = getMobilityTracesMatrix(U2,antenne)
    		dist, cost, path = dwt(class1_sample,class2_sample,False)
    		print "dist L-norm: {0}".format(dist)+","+"{0}".format(U1)+","+"{0}".format(U2)
    #dist, cost, path = dwt(class1_sample,class2_sample,False,dist=euclidean_distance)
    #print "dist Euclidean: {0}".format(dist)
    #dist, cost, path = dwt(class1_sample,class2_sample,True,dist=distance_latlon)
    #print "dist earthmovers: {0}".format(dist)
#end test_dwt_distance


def lcs(class1_sample,class2_sample,eps,gamma):
    """
    Computes the longest commun subsequence in R^2 of two trajectories in space (lat,lon)
    @eps Spatial threshold of lat,lon points proximity
    @gamma represent the number of points the algorithm will look for 
    @class1_sample: user1 matrix representinx a trajectory 
    @class2_sample: user2 matrix representinx a trajectory 
    NOTE: memory problem when trajectories' lengths are very different
    """
    if len(class1_sample) == 0 or len(class2_sample) == 0:
       return -1
    # First property
    m = len(class1_sample)
    n = len(class2_sample)
    #print "{0}<{1},{2}<={3}".format(distance_latlon(class1_sample[-1],class2_sample[-1]),
    #        eps,abs(m-n),gamma)
    if (distance_latlon(class1_sample[-1],class2_sample[-1]) < eps) and abs(m-n)<=gamma:
       return lcs(class1_sample[:-1], class2_sample[:-1],eps,gamma) + 1

    # Second proprerty
    # Last of str1 not needed:
    t1 = lcs(class1_sample[:-1], class2_sample,eps,gamma)
    # Last of str2 is not needed
    t2 = lcs(class1_sample,class2_sample[:-1],eps,gamma)
    
    if (t1) > (t2):
       return t1
    else:
       return t2

#end lcs


def lcss_distance(class1_sample,class2_sample,eps):
    """
    Uses the longest commun subsequence in R^2 of two trajectories in space
    to compute a distance between thses two common subsequences
    @eps Spatial threshold of lat,lon points proximity
    @class1_sample: user1 matrix representinx a trajectory 
    @class2_sample: user2 matrix representinx a trajectory 
    """
    gamma = abs(class1_sample.shape[0]-class2_sample.shape[0])
    gamma = max(class1_sample.shape[0],class2_sample.shape[0])
    gamma = 6
    min_len = float(min(class1_sample.shape[0],class2_sample.shape[0]))
    dist = 1-(lcs(class1_sample,class2_sample,eps,gamma)/min_len)
    return dist
#end lcss_distance


def test_lcs_distance():
    #Getting data
    for i in os.listdir("sample"):
	U1 = "sample/"+i
    	for j in os.listdir("sample"):
		U2 = "sample/"+j
    #U1 = "data/user_1000408639311334.csv"
    #U2 = "data/user_1004631430542939.csv"
    		antenne = getAntennasLocation ()
    		class1_sample = getMobilityTracesMatrix(U1,antenne)
    		class2_sample = getMobilityTracesMatrix(U2,antenne)
    #filterTime(class1_sample,class2_sample)
    		eps = 10 # 1km
		dist=lcss_distance(class1_sample,class1_sample,eps)
    		print "lcss: {0}".format(dist)+","+"{0}".format(U1)+","+"{0}".format(U2)


#end test_dwt_distance

def lcs_string_distance(class1_sample,class2_sample):
    """
    Uses the standar longest commun subsequence in disimilarity function to 
    compute a distance between two trajectories represented as a set of antennas
    @class1_sample: user1 matrix representing a trajectory as a set of antennas(cgis)
    @class2_sample: user2 matrix representing a trajectory as a set of antennas(cgis)
    """
    length, path = mlpy.lcs_std(class1_sample,class2_sample)
    dist = 1-(float(length)/float(min(len(class1_sample),len(class2_sample))))
    return dist
#end lcs_string_distance

def test_lcs_string_distance_iteratively(path,path2):
    """
    Computes LCS string distance between a trajectory and a set of trajectories.
    The trajectories ared ordered by distance
    @path: path of a trajectory
    @path2: path of a set of trajectories
    """
    U1 = path
    resultados=dict()
    for j in os.listdir(path2):
        U2 ="{0}/".format(path2)+ j
        if U1!=U2:
                antenne = getAntennasLocation()
                class1_sample = getMobilityTracesCgi(U1,antenne)
                class2_sample = getMobilityTracesCgi(U2,antenne)
                dist  = lcs_string_distance(class1_sample,class2_sample)
                resultados[dist]=U2
    ord_resultados=collections.OrderedDict(sorted(resultados.items(),reverse=True))

    tray=list()
    for key,value in ord_resultados.iteritems():
        tray.append(value)
    return tray


def test_lcs_string_distance():
    for i in os.listdir("sample"):
	U1="sample/"+i
	for j in os.listdir("sample"):
		U2= "sample/"+j
    #U1 = "data/user_1000408639311334.csv"
    #U2 = "data/user_1004631430542939.csv"
    		antenne = getAntennasLocation ()
    		class1_sample = getMobilityTracesCgi(U1,antenne)
    		class2_sample = getMobilityTracesCgi(U2,antenne)
    		dist = lcs_string_distance(class1_sample,class2_sample)
    		print "lcss: {0}".format(dist)+","+"{0}".format(U1)+","+"{0}".format(U2)
#end test_lcs_string_distance


def distance (u1,u2,typeDistance,minutes=15): 
    d = -1 
    #print "Minutes in distance: {}".format(minutes)
    if (typeDistance == 'lcs_distance'): 
       u1 = u1[:, [0]] 
       u2 = u2[:, [0]] 
    else: 
       u1 = u1[:, [3,1,2] ] 
       u2 = u2[:, [3,1,2] ] 
       #u1 = u1[ [3,1,2],: ] 
       #u2 = u2[ [3,1,2],: ] 

    if (typeDistance == 'pca_distance'): 
       if areTimeOverlaping(u1,u2): 
          (u1a,u2a) = alignInTime( u1, u2,minutes) 
          if (len(u1a)==0) | (len(u2a)==0):
            d = sys.float_info.max 
          else:
            d = pca_distance(u2a,u1a) 
       else: 
          d = sys.float_info.max 
    elif (typeDistance == 'linear_euclidean_distance'): 
       if areTimeOverlaping(u1,u2): 
          (u1a,u2a) = alignInTime( u1, u2, minutes) 
          if (len(u1a)==0) | (len(u2a)==0):
            d = sys.float_info.max 
          else:
            d = linear_distance(u1a,u2a,earthMoverDistance=False) 
       else: 
          d = sys.float_info.max 
    elif (typeDistance == 'linear_earthmover_distance'): 
       if areTimeOverlaping(u1,u2): 
          (u1a,u2a) = alignInTime( u1, u2, minutes) 
          if (len(u1a)==0) | (len(u2a)==0):
            d = sys.float_info.max 
          else:
            d = linear_distance(u1a,u2a,earthMoverDistance=True) 
       else: 
          d = sys.float_info.max 
    elif (typeDistance == 'dwt_distance_earthmovers'): 
        d, cost, path = dwt(u1,u2,dist=distance_latlon)
    elif (typeDistance == 'dwt_distance_euclidean'): 
        d, cost, path = dwt(u1,u2,dist=euclidean_distance) 
    elif (typeDistance == 'lcs_distance'): 
        d = lcs_string_distance(u1.T[0].astype(int),u2.T[0].astype(int)) 
    else: # (typeDistance='dwt_distance'): 
        d, cost, path = dwt(u1,u2,dist=distance_latlon)

    return d 

#end distance 

def _distance( latitude1, longitude1,latitude2, longitude2):
    """ Computes the euclidean distance  in Km"""
    PIov180 = 0.017453292519943295
    dLat = (float(latitude2) - float(latitude1)) * PIov180
    dLon = (float(longitude2) - float(longitude1)) * PIov180
    a = math.sin(dLat/2)**2 + (math.sin(dLon/2)**2) * \
        math.cos(float(latitude1)*PIov180) * \
    math.cos(float(latitude2)*PIov180)

    divisor = math.sqrt(1-a)
    if (divisor == 0):
        divisor = 1
    return round(12742 * math.atan(math.sqrt(a)/divisor), 6)
#end _distance

def _printTrajectory(id,trail_target_imsi,icon):
      """ Print in csv format to be open in a fusion tables to plot points
      """
      i = 0
      row,col = trail_target_imsi.shape
      myformat='%Y-%m-%d %H:%M:%S'
      while i < col:
           print "{},{},{},{},{}".format(id,
             datetime.datetime.fromtimestamp(trail_target_imsi[0][i]).strftime(myformat),
                                            trail_target_imsi[1][i],
                                            trail_target_imsi[2][i],icon)
           i=i+1

def print_results(str,valor_ndcg):
    print "str: {0}".format(valor_ndcg)

def relevances(path,path2):
    """
    Creates a matrix with the results of all 4 distances and computes 
    the relevances of trajectories based in the position on the matrix
    @path: path of a trajectory
    @path2: path of a set of trajectories
    """
    datos = list()
    for i in os.listdir("sample"):
    	datos.append("sample/"+i)
    
    pca = test_pca_distance_iteratively(path,path2)
    linear = test_linear_distance_iteratively(path,path2)
    dtw = test_dtw_distance_iteratively(path,path2)
    lcs = test_lcs_string_distance_iteratively(path,path2)

    matriz = np.array([pca,linear,dtw,lcs]).T
    rel = dict()
    
    for d in datos:
	suma = 0
	for i in range(0,len(matriz.T[0])):
		for j in range(0,len(matriz[0])):
			if d == matriz[i,j]:
				suma = suma+i+1
	rel[d] = suma
    return rel

def relevances_no_lcs():
    """
    Creates a matrix with the results of all distances minus LCS string distance and computes 
    the relevances of trajectories based in the position on the matrix
    @path: path of a trajectory
    @path2: path of a set of trajectories
    """
    datos = list()
    for i in os.listdir("sample"):
	datos.append("sample/"+i)

    pca = test_pca_distance_iteratively()
    linear = test_lienar_distance_iteratively()
    dtw = test_dtw_distance_iteratively()
    
    matriz = np.array([pca,linear,dtw]).T
    rel = dict()
    for d in datos:
	suma = 0
	for i in range(0,len(matriz[0])):
		if d == matriz[0,i]:
			suma = suma+3
	for i in range(0,len(matriz[1])):
		if d == matriz[1,i]:
			suma = suma+2
	for i in range(0,len(matriz[2])):
		if d == matriz[2,i]:
			suma = suma+1
	rel[d] = suma
    return rel

def eval_ranking(lista_trayectorias,path,path2):
    """
    Returns the ndcg of a given ranking
    @lista_trayectorias: ranking of a distance
    @path: path of a trajectory
    @path2: path of a list of trajectories
    """
    dicc_rel = relevances(path,path2)
    lista_relevancias = list()
    for i in range(0,len(lista_trayectorias)):
	lista_relevancias.append(dicc_rel.get(lista_trayectorias[i]))
    #print lista_relevancias
    evaluation = ndcg(lista_relevancias, rank=5)
    return evaluation

def dcg(relevances, rank=10):
    """
    Calculates Discounted cumulative gain (DCG) based on relevances of items
    @relevances: list of relevances of items of a list
    @rank: integer that shows how many relevances should be evaluated
    """
    relevances = np.asarray(relevances)[:rank]
    n_relevances = rank
    score = np.power(2,relevances)-1

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(score / discounts)

def ndcg(relevances, rank=10):
    """
    Calculates Normalized Discounted cumulative gain (nDCG) based on relevances of items
    @relevances: list of relevances of items of a list
    @rank: integer that shows how many relevances should be evaluated
    """
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(relevances, rank) / best_dcg


def calculate(path_tray,carpeta,limite):
    """
    Returns the best ranking of all 4 rankings after evaluating with nDCG 
    and the trajectory that is closer to a initial trajectory.
    @path_tray: path of a trajectory
    @carpeta: path of a directory of trajectories
    @limite: number of trajectories to show
    """
    pca = test_pca_distance_iteratively(path_tray,carpeta)
    linear = test_linear_distance_iteratively(path_tray,carpeta)
    dtw = test_dtw_distance_iteratively(path_tray,carpeta)
    lcs = test_lcs_string_distance_iteratively(path_tray,carpeta)

    resultados = {eval_ranking(pca,path_tray,carpeta):("pca",pca),eval_ranking(linear,path_tray,carpeta):("lin",linear),eval_ranking(dtw,path_tray,carpeta):("dtw",dtw),eval_ranking(lcs,path_tray,carpeta):("lcs",lcs)}

    resultados_ord = collections.OrderedDict(sorted(resultados.items()))
    res_rank = resultados_ord.values()[len(resultados.values())-1]
    res_rank2 = res_rank[1][-int(limite):]
    res_tray = res_rank[1][len(res_rank[1])-1]
    
    for key,value in resultados.iteritems():
	print "NDCG {0}: ".format(value[0])+"{0}".format(key)

    print "Mejor Ranking: {0}".format(res_rank)
    print res_rank2
    print "Trayectoria Escogida: {0}".format(res_tray)


__author__ = "Miguel Nuñez del Prado and Isaias Hoyos"
__copyright__ = "Copyright 2017, Detector Project"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Miguel Nuñez del Prado"
__email__ = "m.nunezdelpradoc@up.edu.pe"
__status__ = "Development"
