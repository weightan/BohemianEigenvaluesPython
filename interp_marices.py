#from bohemian import *
import itertools
from tqdm import tqdm
from math import comb
from numba import jit
import math
from PIL import Image
from numpy.linalg import LinAlgError

from datetime import datetime
import numpy as np
import random 
from matplotlib import cm
from scipy.linalg import circulant

values_for_animation = np.roots([1,0,0,-1])#[1, 0, -1]  

@jit
def log_density_map(val, max_count): 

    brightness = math.log(val) / math.log(max_count)
    gamma = 2.0
    brightness = math.pow(brightness, 1/gamma)

    return brightness




def matrix_interpolate(A, B, iters, rank):

    # A = num_to_matrix(A, rank)
    # B = num_to_matrix(B, rank)

    A = num_to_matrix_withzero(A, rank)
    B = num_to_matrix_withzero(B, rank)

    arr = [(i/iters)*B + (1 - i/iters)*A for i in range(1, iters+1)]
    arr.insert(0, A)
    #arr.append(B)

    # print(arr)
    # print(np.linalg.eigvals(np.array(arr)))

    return np.linalg.eigvals(np.array(arr)).flatten() 

def num_to_matrix(n, rank):

    s = [- 1 if i == '1' else 1 for i in format(n, 'b')]
    s = [1]*(rank**2 - len(s)) + s

    return np.array(s).reshape((rank, rank))
       
def generate_one_comp_m(n, elemets):

    A = np.diag([1 for i in range(n-1)], -1)
    A[:, n-1] = np.random.choice(elemets, (n))

    return A

def generate_one_permut_m(n, elemets):

    p = np.random.choice(elemets, (n))
    A = np.random.permutation(np.diag(p))

    return A    



def generate_one_bohem_m(n, elemets):

    A = np.random.choice(elemets, (n,n))

    return A

def generate_one_tridiag_m(n, elemets):

    rand = np.random.choice(elemets, (n))
    randup = np.random.choice(elemets, (n-1))
    randdown = np.random.choice(elemets, (n-1))
    A = np.diag(rand) + np.diag(randup, 1) + np.diag(randdown, -1)

    return A

def generate_one_circulant_m(n, elemets):

    rand = np.random.choice(elemets, (n))

    return circulant(rand)

def generate_one_vander_m(n, elemets):

    rand = np.random.choice(elemets, (n))

    return np.vander(rand, n)


def mat_batch( rank, iters, elemets ):
    a = [0]*(iters+1)
    # print(a)
    
    # A = generate_one_comp_m(rank, elemets)
    # B = generate_one_comp_m(rank, elemets)

    # A = generate_one_bohem_m(rank, elemets)
    # B = generate_one_bohem_m(rank, elemets)

    A = generate_one_circulant_m(rank, elemets).astype(complex)
    B = generate_one_circulant_m(rank, elemets).astype(complex)



    for i in range( iters+1):
        a[i] =  (i/iters)*B + (1 - i/iters)*A 

        # try:
        #     a[i] = np.linalg.inv( (i/iters)*B + (1 - i/iters)*A ) 

        # except LinAlgError:

        #     a[i] = np.zeros((rank, rank))

        

    # a.insert(0, A)

    # print(a)
    E  = np.linalg.eigvals(np.array(a))

    for i in range( iters+1):

        char_poly = np.poly(np.nan_to_num(E[i], nan=0.0, posinf=10**12, neginf=-10**12)).astype(complex)
        char_poly[-1] = 12**2 
        E[i] = np.roots(char_poly)
        #eigvals = np.roots(np.polyder(char_poly))


    return E.flatten() 



def num_to_matrix_withzero(n, rank):
    arg1 = 1j
    arg2 = 0.1 + 1j
    arg3 = -0.1 + 1j

    newNum = ''
    num = n
    base = 3

    while num > 0:
        newNum = str(num % base) + newNum
        num //= base

    k = newNum
    listK = [0]*len(k)

    for p in range(len(k)):
        # if p == len(k) -1:
        #     listK[p] = complex(100, 0)
        if k[p] == '0':
            listK[p] = arg1
        elif  k[p] == '1':
            listK[p] = arg2
        else:
            listK[p] = arg3


    listK = [0]*(rank**2 - len(listK)) + listK

    return np.array(listK).reshape((rank, rank))


def hist_ordered(generator, rank, N):
    arr = []
    counts = np.zeros((N, N), dtype=np.uint64)
    scale = N*0.01
    iters = 10

    c = 0

    mnums = [i for i in range(1, 2**(rank*2) + 1)]

    for par in tqdm(itertools.combinations(mnums, 2)):

        eigval =  generator(par[0], par[1], iters, rank)

        for v in eigval:
            arr.append(v)

        if len(arr) > 1100:
            for v in arr:

                x = round(v.real*scale + N/2)
                y = round(v.real*scale + N/2)

                if (0 <= x < N) and (0 <= y < N):

                    counts[y, x] += 1
            arr = []

        c += 1
        if c > 100000:
            break


    return counts   #[:, :, 0]


def hist_random(generator, rank, N):
    arr = []
    counts = np.zeros((N, N), dtype=np.uint64)
    scale = N*0.09
    iters = 6000
    samples = 300
    #base = 2
    rank = 4
    elemets = [-1,  0, 1]

    for i in tqdm(range(samples)):

        # eigval =  generator(random.randrange(1, base**(rank*2)),
        #                     random.randrange(1, base**(rank*2)), iters, rank)
        eigval =  generator(rank, iters, elemets)

        for v in eigval:
            arr.append(v)

        if len(arr) > samples // 10:
            for v in arr:

                x = round(v.real*scale + N/2)
                y = round(v.imag*scale + N/2)

                if (0 <= x < N) and (0 <= y < N):

                    counts[y, x] += 1
            arr = []

    return counts


def render_b_vec_hist(N = 2000, rank = 2):

    counts = hist_random(mat_batch, rank, N)

    im_arr = np.zeros((N, N, 3), dtype=np.uint8)

    max_count = np.max(counts)

    bac = (239, 235, 216)
    #bac = (20, 30, 20)

    cmap = cm.get_cmap("copper_r")

    for y in tqdm(range(N)):
        for x in range(N):
            if counts[y, x]:
                rgba = cmap( log_density_map(counts[y, x], max_count) )
                im_arr[y, x, 0] = int(255 * rgba[0])
                im_arr[y, x, 1] = int(255 * rgba[1])
                im_arr[y, x, 2] = int(255 * rgba[2])

                # im_arr[y, x, 0] = 255
                # im_arr[y, x, 1] = 255
                # im_arr[y, x, 2] = 255
            else:
                im_arr[y, x, 0] = bac[0]
                im_arr[y, x, 1] = bac[1]
                im_arr[y, x, 2] = bac[2]

    im = Image.fromarray(im_arr)


    name = f"R_{N}_sampl_{0}_rank_{rank}_{random.random()}.png"

    if N >= 3000:
        name = 'morethan4000/' + name
    else:
        name = 'examples/' + name

    im.save(name)



if __name__ == "__main__":
    # test()
    # values_for_animation = [1, 1j, -1]
    # for i in range(2000):
    #     values_for_animation[2] = values_for_animation[2] *(1 + 0.006j)/abs(1 + 0.006j)
    #     values_for_animation[1] = values_for_animation[1] *(1 - 0.006j)/abs(1 - 0.006j)
    
    render_b_vec_hist(2500)

    # print(np.roots([1,0,0,-1]))

    #companion_mat_batch(4, 5)
    
