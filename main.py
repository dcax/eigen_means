
#This respresents some code to try and create an alternative to k-means for Euclidean clustering.

import numpy as np
from sklearn.linear_model import LinearRegression
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Tests in R^d
d = 2 #dimension
origin = np.zeros(d)
total_points = 10000
expected_clusters = 2
scale = 2.0 #How large each cluster is ;#increase to test
sample_size = 500
sample_number = 100
length = 100
granularity = 1.0E-2

def new_cluster(n=100, center=origin,sigma=1.0):
    #Creates symmetric points about an origin 
    #We desire that a cluster is found to have mean = center
    return np.random.normal(loc=center,scale=sigma,size=(n,d))

def gen_batch():
    #Creates trial data set
    centers = new_cluster(n=expected_clusters, center=origin, sigma=25) #Chooses random centers
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

def gen_unit_line(dim=d+1):
    #generates unit direction lines in dimension dim
    r = np.random.normal(size=(dim,))
    if r[-1] < 0:
        r[-1] = -r[-1]
    return r/np.linalg.norm(r) #length zero incredibly unlikely

def find_force(sample=[], line=np.zeros(d+1),length=length):
    #Finds force on perpindicular swirling item
    #points have mass of 1.
    #function to optimise is abs force
    f = np.zeros(d+1)
    #assumes forces superimpose
    loc = line*length #line is a unit vector usually
    for particle in sample:
        #particle and loc in reference frame of centroid
        #taken with respect to sample
        #trying inverse square (gravitational)
        #f += (particle-loc)/(np.linalg.norm(particle-loc)**3)
        #f += (particle+loc)/(np.linalg.norm(particle+loc)**3)

        #springs
        f += (particle - loc)/np.linalg.norm(particle - loc)**2
        f += (particle + loc)/np.linalg.norm(particle + loc)**2
    return f

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
    fig = plt.figure()
    #Creates plot to eventually use in 3d
    print() 
    for sample in samples:
        #Here is the bulk of hte algorithm
        #We do curve fitting to eigen value of one
        #pprint(sample)
        sample = np.array(sample)
        sample = np.apply_along_axis(lambda v: np.append(v,[0.0]),1,sample)
        #Puts sample on framework of n+1 dimensional array

        centroid = np.sum(sample,axis=0)/sample_size
        print("Sample has centroid {}.".format(centroid))
        #pprint(centroid)
        sample = np.apply_along_axis(lambda s: s-centroid, 1, sample)
        #Sample is now normalised with respect to center

        ax = fig.add_subplot(111, projection='3d')
        #Plotting points
        x = [centroid[0]]
        y = [centroid[1]]
        z = [0]
        for point in sample:
            x.append(point[0])
            y.append(point[1])
            z.append(0) #Already known for just points
        #ax.scatter(x,y,z,marker='o',c='r')
        forces = [] #abs values of the forces
        locx = []
        locy = []
        n = 10000
        for i in range(n):
            
            line = gen_unit_line(d+1)
            #pprint(line)
            locx.append(line[0]*length)
            locy.append(line[1]*length)
            f = find_force(sample,line=line)
            forces.append(np.linalg.norm(f))

        #print("{},{},{}.".format(len(locx),len(locy),len(forces)))
        ax.scatter(locx,locy,forces, marker='^',c='b')

        


        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("|F|")

        plt.show()

        
        exit()







main() #majority of code


