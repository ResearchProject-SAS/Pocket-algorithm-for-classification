import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.utils import shuffle
from matplotlib import style
style.use("ggplot")
from collections import OrderedDict
from sklearn.datasets import make_circles 
from mpl_toolkits.mplot3d import Axes3D 
  
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def y(x, m, b):
    return m*x + b

X = np.array(np.linspace(0, 6, 2400))
y_above = np.array([y(x, 10, 5) + abs(random.gauss(20,8)) for x in X])
y_below = np.array([y(x, 10, 5) - abs(random.gauss(20,8)) for x in X])

x0=np.stack((X,y_above),axis=1)
x0=np.hstack((x0, np.zeros((x0.shape[0], 1), dtype=x0.dtype)))
x1=np.stack((X,y_below),axis=1)
x1=np.hstack((x1, np.ones((x1.shape[0], 1), dtype=x1.dtype)))
ds=np.concatenate((x0,x1))

n = np.array(np.linspace(0, 6, 100))
y_above = np.array([y(x, 10, 5) + abs(random.gauss(4,8)) for x in n])
y_below = np.array([y(x, 10, 5) - abs(random.gauss(4,8)) for x in n])

noise_x=np.stack((n,y_below),axis=1)
noise_x=np.hstack((noise_x, np.zeros((noise_x.shape[0], 1), dtype=noise_x.dtype)))

noise_y=np.stack((n,y_above),axis=1)
noise_y=np.hstack((noise_y, np.ones((noise_y.shape[0], 1), dtype=noise_y.dtype)))
noise=np.concatenate((noise_x,noise_y))

ds=np.concatenate((ds,noise))
dataset=pd.DataFrame(ds) 
x=dataset.iloc[:,0:2].values
y=(dataset.iloc[:,2].values).astype(int)

plt.figure(figsize=(10,8))
plt.plot(X,10*X+5,color='black')
plt.title("Non-Linear Dataset")
plt.scatter(x[y==0,0],x[y==0,1],x[y],c='r',alpha=1)
plt.scatter(x[y==1,0],x[y==1,1],x[y],c='g')
plt.show()

x, y = shuffle(x, y)

weights = np.random.uniform(low=0.0, high=0.4, size=2)
def pocketAlgorithm(x, weight, y):
    iterations = 50
    alpha = 0.01
    constraint = 1
    w = weight
    

    brokenConstraints = []
    #While there exist a constraint run the PLA
    while iterations > 0:
        constraint = 0
        #Randomly choose indexes from our data x
        for i in np.random.permutation(range(len(x))):
            
            y_calc = np.dot(w, x[i])
            #Check wheter w*x violates a constraint
            #yc=y_calc
            if(y_calc < 0 and y[i] == 1):
                constraint += 1
                w = np.add(w, alpha*x[i])
                
            elif(y_calc > 0 and y[i] == -1):
                constraint += 1
                w = np.subtract(w, alpha*x[i])
        
        brokenConstraints.append(constraint)
        iterations -= 1
    weights=w
    return brokenConstraints

#Plots Misclassified points vs 7000 Iterations
def plot_missclass(missMat, maxIter):
    #plt.figure(figsize=(15,2))
    plt.plot(missMat)
    # plt.xticks(np.arange(0, maxIter, 100))
    plt.xlim(0,maxIter)
    plt.xlabel('Iterations')
    plt.ylabel('Misclassifications')
    plt.title('Misclassified points vs Iterations of Pocket Algorithm')
    plt.show()

brokenConstraints = pocketAlgorithm(x, weights, y)
print('Plotting...')
#print(brokenConstraints)
plot_missclass(brokenConstraints, 70)
#print(weights)
pkaRes=pd.DataFrame([weights],columns=['w0','w1'])
print(pkaRes)

#plot the image
axisX=range(len(brokenConstraints))
#print(len(brokenConstraints))
fig=plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('Pocket Algorithm figure')
plt.xlabel('Times')
plt.ylabel(' points')
ax1.scatter(axisX,brokenConstraints,c = 'r',marker = '.')
plt.legend('x')
plt.show()