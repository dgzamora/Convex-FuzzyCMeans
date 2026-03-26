##Import data

import pandas as pd
import os
import kagglehub


path = kagglehub.dataset_download("muhammadkhubaibahmad/student-performance-and-clustering-dataset")
print("Path to dataset files:", path)


# List files in the dataset
files = os.listdir(path)
print("Files available:", files)

# Load the main dataset (adjust filename if necessary)
df = pd.read_csv(os.path.join(path, 'student_dropout_behavior_dataset.csv'))  # check actual filename from printed list
print(df.head())
print(df.info())

import matplotlib.pyplot as plt
column="quiz1_marks"
# Replace 'column_name' with your actual column name
plt.hist(df[column], bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of '+column)
plt.show()




##Main algorithm
import numpy as np
from numpy import linalg as LA

def phi(x,alpha):
    return alpha*x**2+(1-alpha)*x

def distancia(x,center,index,Bounds,alpha=0):
    c=center.copy()
    c=np.append(c,Bounds[0])
    c=np.append(c,Bounds[1])
    c.sort()
    M=Bounds[1]-Bounds[0]
    dis=100
    j=index+1
    if(c[j]<=x<=c[j+1]):
        dis=phi((x-c[j])/(c[j+1]-c[j]),alpha)
    elif(c[j-1]<=x<=c[j]):
        dis=phi((c[j]-x)/(c[j]-c[j-1]),alpha)
    return dis

def distanciaE(x,center):
    return (x-center)**2

def acota(x,c): #returns the partition in which x is (there are k+1 partitions)
    c = np.asarray(c)
    k = c.shape[0]
    c.sort()
    index=k
    for j in range(k):
        if (c[j]>x):
            index=j
            break
    return index


def updateMF(x,c,m=2): #we can include alpha=0
    x = np.asarray(x)
    c = np.asarray(c)
    c.sort()
    x.sort()
    n = x.shape[0]
    k = c.shape[0]
    d = np.zeros((n,k))
    Bounds=[np.min(x),np.max(x)]

    for i in range(n):
        for j in range(k):
            d[i,j]=distanciaE(x[i],c[j])
            #d[i,j]=distancia2(x[i],c,j,Bounds,alpha)

    u=np.zeros((n,k))
    for i in range(n):
        partition=acota(x[i],c)
        if (partition==0):
            u[i,0]=1
            continue
        elif(partition==k):
            u[i,k-1]=1
            continue
        elif(d[i,partition-1]==0):
            u[i,partition-1]=1
            continue
        elif(d[i,partition]==0):
            u[i,partition]=1
            continue
        else:
            u[i,partition-1]=1/(1+pow(d[i,partition-1]/d[i,partition],1/(m-1)))
            u[i,partition]=1/(1+pow(d[i,partition]/d[i,partition-1],1/(m-1)))
    return u

def updateCenter(x,u,m):
    x = np.asarray(x)
    um = np.asarray(u)**m
    n = u.shape[0]
    k = u.shape[1]
    c=np.zeros(k)
    for j in range(k):
        c[j]=np.dot(x,um[:,j])/np.sum(um[:,j])
    return c

def initialize(x,k):
    x = np.asarray(x)
    Bounds=[np.min(x),np.max(x)]
    c=np.zeros(k)
    for j in range(k):
        #c[j]=np.percentile(x,(j+1)/(k+1)*100)
        c[j]=Bounds[0]+(Bounds[1]-Bounds[0])*(j+1)/(k+1)
    return c

def ConvexFKMeans(x,k,m=2,tol=0.001,maxiter=10): #we can include alpha=0
    x = np.asarray(x)
    x.sort()
    n = x.shape[0]
    c=initialize(x,k)
    u=updateMF(x,c,m) #we can include alpha=0
    error=1
    iter=1
    while(error>=tol and iter<=maxiter):
        cnew=updateCenter(x,u,m)
        unew=updateMF(x,cnew,m) #we can include alpha=0
        error=LA.norm(c-cnew)
        iter=iter+1
        u=np.copy(unew)
        c=np.copy(cnew)
    return c,u, error, iter

def core(x,u,tol=0.01):
    u = np.asarray(u)
    core=x[np.where(u>=1-tol)[0]]
    return [np.min(core),np.max(core)]
def supp(x,u,tol=0.01):
    u = np.asarray(u)
    supp=x[np.where(u>=tol)[0]]
    return [np.min(supp),np.max(supp)]
def coreBounds(x,u, tol=0.01):
    x = np.asarray(x)
    u = np.asarray(u)
    k=u.shape[1]
    c=np.zeros(2*k)
    for j in range(k):
        interval=core(x,u[:,j],tol=tol)
        c[2*j]=interval[0]
        c[2*j+1]=interval[1]
    return c
def updateMF_fromcore(x,cores,m=2):
    x = np.asarray(x)
    cores = np.asarray(cores)
    cores.sort()
    x.sort()
    n = x.shape[0]
    k =int(cores.shape[0]/2)
    Bounds=[np.min(x),np.max(x)]
    u=np.zeros((n,k))

    for i in range(n):
        index=2*k-2
        for j in range(2*k-1):
            if(cores[j]<=x[i]<cores[j+1]):
                index=j
                break

        if (index %2==0):
            j=int((index)/2)
            u[i,j]=1
            continue
        else:
            j=int((index-1)/2)
            d1=distanciaE(x[i],cores[index])
            d2=distanciaE(x[i],cores[index+1])
            if(d1==0):
                u[i,j]=1
            elif(d2==0):
                u[i,j+1]=1
            else:
                u[i,j]=1/(1+pow(d1/d2,1/(m-1)))
                u[i,j+1]=1/(1+pow(d2/d1,1/(m-1)))

    return u

## Results
# Extract column as a NumPy array
x = df[column].dropna().values
x = np.sort(x)

# Number of clusters (e.g., 3)
k = 5
tol=0.001
c,u,error, iter=ConvexFKMeans(x,k,maxiter=100)

# Plot each cluster’s membership function
plt.figure(figsize=(10, 5))
for j in range(u.shape[1]):
    plt.plot(x, u[:, j], label=f'Cluster {j+1}')
print("Centers:\t",c)
plt.title('Convex Fuzzy K-Means Membership Functions')
plt.xlabel('x')
plt.ylabel('Membership degree')
plt.legend()
plt.grid(True)
plt.show()