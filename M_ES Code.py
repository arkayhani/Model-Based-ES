
import math
import random
import numpy as np
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt

from scipy import *
from scipy.linalg import norm, pinv
from sklearn.cluster import KMeans 
from matplotlib import pyplot as plt

N=10 

class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 0.5
        self.W = random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
 
        assert len(d) == self.indim
        return np.exp(-np.multiply(self.beta ,np.power(np.divide(norm(c-d),norm(d)),2)))
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        #rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        #self.centers = [X[i,:] for i in rnd_idx]
        
        #print(self.centers)
        X=np.array(X)
        Y=np.array(Y) 
        kmeans = KMeans(n_clusters=min(self.numCenters,X.shape[0]), random_state=0).fit(X)
        self.centers=kmeans.cluster_centers_
        
        
        #print(self.centers)
        #print "center", self.centers
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print G
        
        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
        X=np.array(X)
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y
        

landa=1
def sphere(x):
    sum=0
    for i in range(N):
        sum=sum+math.pow(x[i],2)
    return sum
def normalI():
    no=np.random.normal(0,np.ones(N))
    return no
def normal():
    no=np.random.normal(0,1)
    return no
numT=1
kl10=[]
#numeber of real generation
numevals=1000
#numeber of traials
numr=5
#numeber of model generation
nummodels=0
#numeber of training points
numtraining=30
#nhidden layer
numhidden=20
##KLs are for recording the results in plots
kl3=np.zeros(numevals)
kl4=np.zeros(numevals-2)
kl5=np.zeros(numevals)
kl6=np.zeros(numevals)
kl18=[]
for ss in range(numr):
    
    ######################3
    kk=np.zeros(numevals) 
    kk3=[]
    kl10=[]
    slop=[]
    modelscount=[]
    firstfit=0
    
    #kk2=np.zeros(3000) 
    turns=0
    for numT in range(1):
        eval=0
        tou=1.0/(math.pow(N,0.5))
        toui=1.0/(math.pow(N,0.25))
        sigma=1
        sigma=sigma
       
        
    
        ind=[]
    
        for j in range(N):
         ind.append(float(random.randint(0,1024)-512)/100)
        #print(ind)
        pop=[]
        pop2=[]
        
        beta=0
        print("p2"+str(turns)+"ss:"+str(ss))
        MSE=0
    

    
    
    
        while(eval<numevals):
            
            #ind2=ind+np.multiply(sigma,normalI())
            print("p2"+str(turns)+"ss:"+str(ss)+"eval:"+str(eval))
            sigma2=sigma
            flag2=0
           
            ind2=ind

            if(pop.__len__()>1 and nummodels>0):
                
                for models in range(nummodels):
                    #if(MSE>sphere(ind2) ):
                     #  break
                    ind3=ind2+np.multiply(sigma2,normalI())
                  
                    
                    
                   
                
                    s3=rbf.test(np.array([ind3]))
                    print("error: "+str(math.fabs(sphere(ind3))))
                    s2=rbf.test(np.array([ind2]))
                    flag=0
                    if(s2>s3):
                        ind2=ind3
                        flag=1
                        flag2=1
                    sigma2=np.multiply(sigma2,np.power(np.exp(flag-0.5),float(1.0/N)))
                MSE=math.fabs(rbf.test(np.array([ind2]))-sphere(ind2))
                kl10.append(MSE/eval)  
            else:
                ind2=ind+np.multiply(sigma,normalI())
                if(pop.__len__()>1):
                    MSE=math.fabs(rbf.test(np.array([ind2]))-sphere(ind2))
                    kl10.append(MSE/eval)  

                     
#            
            newfitness=sphere(ind2)

            modelscount.append(nummodels) 
            flag=0    
            if(eval==0 or newfitness<bestfitness ):
                    bestfitness=newfitness
                    ind=ind2
                    flag=1
            if (eval==0):
                        firstfit=newfitness
#                    
                    print(nummodels)
            if(nummodels>1):
                
                sigma=sigma2
            else:
                sigma=np.multiply(sigma,np.power(np.exp(flag-0.2),float(1.0/N)))


                    
                
            
            pop2.append(newfitness)
              
            pop.append(ind2)
            
            rbf=RBF(N,numhidden,1)
 
            rbf.train(pop,pop2)

               

            eval+=1
    
            if(pop.__len__()>numtraining ):
               
                
                pop2.remove(pop2[0])
                pop.remove(pop[0])
               
        
            kk3.append(bestfitness)
            slop.append((firstfit-bestfitness)/float(eval))
            print(bestfitness)
        turns+=1

    
    ####################################
    
    
    
    
    
    


    kl3=kl3+kk3
    kl4=kl4+kl10
    kl5=kl5+slop
    kl6=kl6+modelscount


kl3/=numr
kl4/=numr
kl5/=numr
kl6/=numr
#plt.figure(1)
#plt.plot(np.log10(kk3),"b--")
plt.figure(1)
plt.plot(np.log10(kl3),"r--")
plt.xlabel("Original Fitness Generation")
plt.ylabel("Original Fitness(log10)")
plt.figure(2)
plt.plot(np.log10(kl4),"g--")
plt.xlabel("Original Fitness Generation")
plt.ylabel("Model Fitness Error(log10)")
plt.figure(3)
plt.plot(np.log10(kl5),"b--")
plt.xlabel("Original Fitness Generation")
plt.ylabel("Original Fitness Slope(log10)")
plt.figure(4)
plt.plot(kl6,"y--")
plt.xlabel("Original Fitness Generation")
plt.ylabel("Model Fitness Generation")
f = open('fiA10.txt', 'w')

for inx2 in range(0,len(kl3)):
    f.write(str(np.log10(kl3[inx2]))+'\n')  # python will convert \n to os.linesep
  # python will convert \n to os.linesep

f.close()
f = open('fierrorA10.txt', 'w')
for inx2 in range(0,len(kl4)):
    f.write(str(np.log10(kl4[inx2]))+'\n')  # python will convert \n to os.linesep
  # python will convert \n to os.linesep

f.close()
f = open('fislopeA10.txt', 'w')
for inx2 in range(0,len(kl5)):
    f.write(str(np.log10(kl5[inx2]))+'\n')  # python will convert \n to os.linesep
  # python will convert \n to os.linesep

f.close()
f = open('fimodelA10.txt', 'w')
for inx2 in range(0,len(kl6)):
    f.write(str(kl6[inx2])+'\n')  # python will convert \n to os.linesep
  # python will convert \n to os.linesep

f.close()
