
#This respresents some code to try and create an alternative to k-means for Euclidean clustering.

import numpy as np
from sklearn.linear_model import LinearRegression
from pprint import pprint

#Tests in R^3
origin = np.zeros(3)
total_points = 10000
expected_clusters = 3
scale = 2.0 #How large each cluster is ;#increase to test
sample_size = 500
sample_number = 100

def new_cluster(n=100, center=origin,sigma=1.0):
    #Creates symmetric points about an origin 
    #We desire that a cluster is found to have mean = center
    return np.random.normal(loc=center,scale=sigma,size=(n,3))

def gen_batch():
    #Creates trial data set
    centers = new_cluster(n=expected_clusters, center=origin, sigma=100) #Chooses random centers
    #print(centers.shape)
    r = []
    clusters = [] #contains known cluster data for comparison
    for i, c in enumerate(centers):
        size = np.random.geometric(p=expected_clusters/total_points)
        #sises will not add to total exactly but will in limiting case
        cluster = new_cluster(n=size, center=c, sigma = scale)
        r.extend(cluster)
        
        #print(i)
        clusters.append(cluster)
    np.random.shuffle(r)
    return (clusters, r)

def get_samples(clump = [], sample_number=sample_number, sample_size=sample_size):
    #gets random samples for monte carlo usage
    r = []
    indices = np.asarray(range(len(clump)))
    for i in range(sample_number):
        np.random.shuffle(indices)
        #pprint(indices)
        r.append([])
        for index in indices[0:sample_size]:
            #get first sample_size random indices
            #pprint(clump[i])
            r[i].append(clump[index])
        #pprint(r[i])
    return r


def main():
    clusters, clump = gen_batch()

    #We now operate on the clump
    #print(clump)
    #print(clusters)
    

    samples = get_samples(clump, sample_number=sample_number, sample_size=sample_size)
    #pprint(samples[0])
    clump = np.array(clump)

    #First we print expected means
    for index, cluster in enumerate(clusters):
        mean = np.sum(cluster,axis=0)/len(cluster)
        print("Cluster #{} at mean {}.".format(index, mean))

    print() 
    for sample in samples:
        #Here is the bulk of hte algorithm
        #We do curve fitting to eigen value of one
        #pprint(sample)
        sample = np.array(sample)
        centroid = np.sum(sample,axis=0)/sample_size
        print("Sample has centroid {}.".format(centroid))
        #pprint(centroid)
        sample = np.apply_along_axis(lambda s: s-centroid, 1, sample)
        #Sample is now normalised with respect to center
        reg = LinearRegression().fit(sample, np.apply_along_axis(lambda s: s*np.linalg.norm(s),1,sample))
        pprint(reg.score(sample, sample))
        #print(reg.intercept_)
        #Intercept is so small as to be essentially insignifigant
        pprint(reg.coef_)
        A = reg.coef_
        eig = np.linalg.eig(A)
        pprint(eig) #This is the meat of what we want
        for eig_vec in eig[1]:
            print("Founded eigen vector {}.".format(eig_vec + centroid))
        exit()

    





main() #majority of code


