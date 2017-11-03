import numpy as np
import matplotlib.pyplot as plt
import math
#
def calc_euc_distance(v1, v2):
    return math.sqrt(math.pow(v1[0] - v2[0],2) + math.pow(v1[1] - v2[1],2))
#
def get_closest_prototype(feature):
    """ Calculates euclidean distance to closest prototype, and returns said prototype """
    min_distance = 100000
    index = 0
    for i, prototype in enumerate(prototypes):
        euc_d = calc_euc_distance(feature,prototype)
        if min_distance > euc_d:
            min_distance = euc_d
            index = i
    return index, min_distance
#
prototypes = np.array([[.25,.25],[.5,.5],[.75,.75]])
M_g = len(prototypes)
sigma=0.095
number=50
T = 20
rate = lambda t : 0.01 * np.exp(-.1*t)
#
# storing the centroid index (note this may not correspond to teh same number from the k-means algorithm)
X11=np.concatenate((sigma*np.random.randn(number,2)+prototypes[0],np.full((number,1),0.0)),axis=1)
X22=np.concatenate((sigma*np.random.randn(number,2)+prototypes[1],np.full((number,1),1.0)),axis=1)
X33=np.concatenate((sigma*np.random.randn(number,2)+prototypes[2],np.full((number,1),2.0)),axis=1)
X = np.concatenate((X11,X22,X33), axis=0)
np.random.shuffle(X)
#
# plt.figure()
# col={0:'bo',1:'go',2:'co'}
# for i in range(0,len(X[:,0])):
#     plt.plot(X[i,0],X[i,1],col[int(X[i,2])])
# plt.plot(prototypes[:,0],prototypes[:,1],'ro')
# plt.axis([0,1.0,0,1.0])
# plt.show()
#
split = int((number * M_g) * .7)
x_train = X[0:split,:]
x_test = X[split:,:]
#
plt.figure()
col={0:'bo',1:'go',2:'co'}
for i in range(0,len(x_train[:,0])):
    plt.plot(x_train[i,0],x_train[i,1],col[int(x_train[i,2])])
plt.plot(prototypes[:,0],prototypes[:,1],'ro')
plt.axis([0,1.0,0,1.0])
plt.show()
for i in range(T):
    for feature in x_train:
        index, euc_d = get_closest_prototype(feature=feature)
        if index == feature[2]:
            m_i = prototypes[index] + rate(t=i) * (feature[0:1] - prototypes[index])
        else:
            m_i = prototypes[index] - rate(t=i) * (feature[0:1] - prototypes[index])
        #
        prototypes[index] = m_i
        #
    #
    print(prototypes)
    plt.figure()
    for j in range(0,len(x_train[:,0])):
        plt.plot(x_train[j,0],x_train[j,1],col[int(x_train[j,2])])
    plt.plot(prototypes[:, 0], prototypes[:, 1], 'ro')
    plt.axis([0, 1.0, 0, 1.0])
    plt.show()