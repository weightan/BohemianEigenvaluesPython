#import itertools 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import math
from numpy import linalg as LA
from random import random, randrange, choice
from tqdm import tqdm

Mn = 5
N = 2000
cmap = 'hot'
inch = 5
iterationsN = 100_000
maxV = 40
pnum = 3

scale = N/8.5
disp = (N/2, N/2)

k0 = 0 
k1 = 1
p0 = 0


"""
A = (5 + 8j)*random() - (3 + 5j)
B = (5 + 8j)*random() - (3 + 5j)

M = np.array([[1, 0, 0, 1], 
            [0, 0, A, 1],
            [1, 0, -1, 1],
            [1, B, 0, 1]])

M = np.array([[A, 0, 1, 1], 
            [0, 1, A, 0],
            [1, 0, 1, 1],
            [1, B, 0, 0]])

M = np.array([[1, 1, 0, 1], 
            [B, 0, 1, A],
            [1, 1, A, 1],
            [1, 0, B, -1]])

M = np.array([[1, 1, 0, 1], 
            [B, 0, 1, A],
            [1, 1, A, 1],
            [1, 0, B, -1]])
"""

def tridiag(n, k1=-1, k2=0, k3=1):

    a = np.empty((n-1))
    b = np.empty((n))
    c = np.empty((n-1))

    for i in range(n-1):
        a[i] = choice([1,-1])
        b[i] = choice([1,-1])
        c[i] = choice([1,-1])

    b[n-1] =  choice([1,-1])

    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def tridiag2(n, k1=-1, k2=0, k3=1):

    a = np.empty((n-1), dtype = np.complex64)
    b = np.empty((n), dtype = np.complex64)
    c = np.empty((n-1), dtype = np.complex64)

    for i in range(n-1):
        a[i] = choice([1,-1, 0])
        b[i] = choice([1,-1, 0])*1j
        c[i] = choice([1,-1, 0])

    b[n-1] =  choice([1,-1])*1j

    return np.diag(a, k1, ) + np.diag(b, k2) + np.diag(c, k3)

def makeParamsR(n, paramasArr):
    g = [paramasArr[i][0] * random() - paramasArr[i][1] for i in range(n)]
    return g

def run(iter1):

    #imre = [randrange(0, 10) +  randrange(0, 10) *1j, randrange(0, 10) +  randrange(0, 10) *1j]
    #imre = [3j - k0, 2j- k1]
    #paramasArr = [imre for i in range(pnum)]
    

    coef = np.zeros((N, N), dtype = np.int32)
    
    for  i in range(iterationsN):

        A = (5 + 8j)*random() - (3 + 5j)*k0
        B = (5 + 8j)*random() - (3 + 5j)*k1

        M = np.array([[1, 0, -1, 1], 
                      [0, 0, A, 1],
                      [1, -1, 1, p0],
                      [B, 0, 1, 0]])
        valArr, tr = LA.eig(M)

        for j in valArr:
            if np.imag( j ) != 0:
                x = round(np.imag( j ) * scale +  disp[0] )
                y = round(np.real( j ) * scale +  disp[1] )

                #if x < N and x > 0 and y < N and y > 0 and coef[x, y] < 40:
                if  x < N and y > 0 and x > 0 and y < N and  coef[x, y] < maxV:
                    coef[x, y] += 1

    #print(imre)
    #coef = np.rot90(coef)

    filenameArr = f'N_{N}_Mn_{Mn}_iterationsN_{iterationsN}'
    #np.save(filenameArr, coef)

    

    for i in range(N):
        for j in range(N):
            if coef[i, j]:
                coef[i, j] += 200
     
    

    plt.figure(num = None, figsize=(inch, inch), dpi=300)

    plt.axis('off')

    plot = plt.imshow(coef, cmap = cmap )

    #### , interpolation='lanczos'

    filenameImage = f'F{iter1}_N_{N}_cmap_{cmap}_Mn_{Mn}_iterationsN_{iterationsN}_{randrange(100_000_000, 1_000_000_000)}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight', pad_inches=0.0)

    ####

    #plt.show()
    plt.close()







if __name__ == '__main__':
    print('start')

    frames = 1200
    step = 0.008

    for i in range (frames):

        k0 += step
        p0 += step

        run(i)
        print(f"done {i}")
    
    
