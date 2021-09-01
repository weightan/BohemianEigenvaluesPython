import numpy as np
from scipy import linalg
import random

from numpy.linalg import LinAlgError
from tqdm import tqdm

from PIL import Image
from matplotlib import cm

import matplotlib.pyplot as plt

from numba import jit

from datetime import datetime

import math

from scipy.linalg import circulant
#  Here's my take on the implementation of polyeig in python.
#  As a test case, I used the same example as the one provided in Mathworks.

#  scaling: The eigenvectors are not the same as the ones returned by Matlab (since they are defined to a scaling factor).
#  I'm not sure what normalization Matlab uses, so in that case I simply normalized by the maximum value of each vector.

# sorting: Matlab does not sort the eigenvalues.
#  In that example, scipy seems to return the eigenvalues in the same order as Matlab.
#   You can uncomment the lines in the script to sort the eigenvalues.

def generate_one_tridiag_m(n, elemets):

    rand = np.random.choice(elemets, (n))
    randup = np.random.choice(elemets, (n-1))
    randdown = np.random.choice(elemets, (n-1))
    A = np.diag(rand) + np.diag(randup, 1) + np.diag(randdown, -1)

    return A

def generate_one_bohem_m(n, elemets):

    A = np.random.choice(elemets, (n,n))


    return A

def generate_one_circulant_m(n, elemets):

    rand = np.random.choice(elemets, (n))

    return circulant(rand)

def generate_one_comp_m(n, elemets):

    A = np.diag([1 for i in range(n-1)], -1)
    A[n-1, :] = np.random.choice(elemets, (n))

    #A[:, n-1] = 2*np.random.beta(0.01, 0.01, (n)) - 1
    return A

def generate_one_comp2_m(n, elemets):

    A = np.diag([complex(1, 0) for i in range(n-1)], -1)
    A[:, n-1] = np.random.choice(elemets, (n))
    A[0, :] = np.random.choice(elemets, (n))

    # A[:, n-1] = 2*np.random.beta(0.01, 0.01, (n)) - 1
    # A[0, :] =2*np.random.beta(0.01, 0.01, (n)) - 1

    return A

def generate_one_vander_m(n, elemets):

    A = np.random.choice(elemets, (n))
    #A = 2*np.random.beta(0.01, 0.01, (n)) - 1

    return np.vander(A, n)

def generate_one_bohem_m_beta(n, elemets):

    A = 2*np.random.beta(0.01, 0.01, (n,n)) - 1

    return A    

def generate_diag(n, elemets):

    rand = np.random.choice(elemets, (n))
    A = np.diag(rand)

    return A    

def random_checkerboard_matrix (n, elemets):
    A  = np.zeros((n,n), dtype = complex)
    n_odd = n - 2 if n % 2 == 1 else n - 1

    for i in range(-n_odd, n_odd + 1, 2):
        #v = np.random.choice(elemets, ( n - abs(i)))
        v = np.random.beta(0.01, 0.01, ( n - abs(i)))
        A += np.diag(v,  k=i)

    return A

@jit
def log_density_map(val, max_count): 

    brightness = math.log(val) / math.log(max_count)
    gamma = 2.2
    brightness = math.pow(brightness, 1/gamma)

    return brightness

@jit
def complex_to_image(z, width, height, real_range, imag_range, centered_at):
    bottom_left = centered_at - complex(real_range / 2, imag_range / 2)
    # Find x coordinate
    x = (z.real - bottom_left.real) / real_range # Normalized
    x *= width

    # Find y coordinate
    y = (z.imag - bottom_left.imag) / imag_range # Normalized
    y = 1 - y
    y *= height

    return (int(x), int(y))

def generate_one_circulant_m_beta(n, elemets):

    #rand = np.random.choice(elemets, (n))
    rand = 2*np.random.beta(0.01, 0.01, (n)) - 1

    return circulant(rand)

def polyeig(*A):
    """
    Solve the polynomial eigenvalue problem:
        (A0 + e A1 +...+  e**p Ap)x=0 

    Return the eigenvectors [x_i] and eigenvalues [e_i] that are solutions.

    Usage:
        X,e = polyeig(A0,A1,..,Ap)

    Most common usage, to solve a second order system: (K + C e + M e**2) x =0
        X,e = polyeig(K,C,M)

    """
    if len(A) <=0:
        raise Exception('Provide at least one matrix')
    for Ai in A:
        if Ai.shape[0] != Ai.shape[1]:
            raise Exception('Matrices must be square')
        if Ai.shape != A[0].shape:
            raise Exception('All matrices must have the same shapes');

    n = A[0].shape[0]
    l = len(A)-1 

    # Assemble matrices for generalized problem
    C = np.block([
        [np.zeros((n*(l-1),n)), np.eye(n*(l-1))],
        [-np.column_stack( A[0:-1])]
        ])

    D = np.block([
        [np.eye(n*(l-1)), np.zeros((n*(l-1), n))],
        [np.zeros((n, n*(l-1))), A[-1]          ]
        ]);

    # Solve generalized eigenvalue problem
    e, X = linalg.eig(C, D);
    if np.all(np.isreal(e)):
        e=np.real(e)
    X=X[:n,:]

    # Sort eigenvalues/vectors
    #I = np.argsort(e)
    #X = X[:,I]
    #e = e[I]

    # Scaling each mode by max
    X /= np.tile(np.max(np.abs(X),axis=0), (n,1))

    return X, e

def test0():
    M = np.diag([3, 1, 3, 1])

    C = np.array([[0.4 , 0 , -0.3 , 0],
                  [0 ,   0 ,  0 ,   0],
                  [-0.3 , 0 ,  0.5 , -0.2 ],
                  [ 0 ,  0 , -0.2 , 0.2]])

    K = np.array([[-7, 2 , 4 , 0],
                  [2 , -4 , 2, 0],
                  [4 , 2 , -9, 3],
                  [0 , 0 , 3, -3]])

    X, e = polyeig(K, C, M)

    print('X:\n',X)
    print('e:\n',e)

    # Test that first eigenvector and value satisfy eigenvalue problem:

    s = e[0];
    x = X[:,0];

    res = (M*s**2 + C*s + K).dot(x) # residuals

    assert(np.all(np.abs(res)<1e-12))

def test1():
    M = generate_one_bohem_m(3, [-1, 0, 1])

    C = generate_one_bohem_m(3, [-1, 0, 1])

    K = generate_one_bohem_m(3, [-1, 0, 1])

    X, e = polyeig(K, C, M)

    print('X:\n',X)
    print('e:\n',e)


def bohem_classic(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n =  4 # n x n matrix
    mono_coloring = False
    counts = np.zeros((width, height), dtype=np.uint64)

    #values = np.array([complex(-1, 0), complex(0, 0), complex(1,0) ])

    #values = np.roots([1,0,0,0,0, -1])

    values = np.array([ complex(0, 1),
                        complex(1, 0), complex(-1, -1)]) 
                        #complex(0, 1), complex(0, -1),
                        #complex(0.01, 0), complex(-0.01, 0),
                        #complex(0, 0.01), complex(0, -0.01)])

    for _ in tqdm(range(samples)):

        mat0 = generate_one_circulant_m_beta(n, values)
        mat1 = generate_one_circulant_m_beta(n, values)
        mat2 = generate_one_circulant_m_beta(n, values) 

        # rand0 = np.random.choice(values, (n))
        # rand1 = np.random.choice(values, (n))
        # rand2 = np.random.choice(values, (n))

        # mat0 = np.diag(rand0)
        # mat1 = np.diag(rand1)
        # mat2 = np.diag(rand2)

        try:
            _, eigvals = polyeig(mat0, mat1, mat2)

        except LinAlgError:
            pass
            #print("Eigenvalue error")

        char_poly = np.poly(np.nan_to_num(eigvals, nan=0.0, posinf=10**12, neginf=-10**12))
        char_poly[-1] = (10**4)
        eigvals = np.roots(char_poly)

        for z in eigvals:
            if abs(z.imag) > 0.0000001:
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
                if (0 <= x < width) and (0 <= y < height):
                    counts[y, x] += 1

    print("Done!")

    max_count = np.max(counts)
    print("Max count:", max_count)

    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x]:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else: 
                    rgba = cmap( log_density_map(counts[y, x], max_count) )
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])
            # else:
            #     rgba = cmap( 0 )
            #     im_arr[y, x, 0] = int(255 * rgba[0])
            #     im_arr[y, x, 1] = int(255 * rgba[1])
            #     im_arr[y, x, 2] = int(255 * rgba[2])



    im = Image.fromarray(im_arr)
    return im


def run_big_picture():
    # Size of complex window w/ respect to a center focal point
    centered_at = complex(0, 0)

    r = 1
    
    real_offset = (centered_at.real - r, centered_at.real + r)
    imag_offset = (centered_at.imag - r, centered_at.imag + r)

    real_range = real_offset[1] - real_offset[0]
    imag_range = imag_offset[1] - imag_offset[0]

    scale = 9 #3.6 2.5

    real_range *= scale
    imag_range *= scale

    samples = 5_0_000 
    
    aspect_ratio = real_range / imag_range
    width = 2500
    height = int(width * 1 / aspect_ratio)

    
    
    # Initialize color map
    # cmap = cm.get_cmap("jet")
    # cmap = cm.get_cmap("viridis")
    # cmap = cm.get_cmap("nipy_spectral")
    cmap = cm.get_cmap("hot")

    
    #im = littlewood(im_arr, width, height, real_range, imag_range, centered_at, cmap)

 
    im_arr = np.zeros((height, width, 3), dtype=np.uint8)
    im = bohem_classic(im_arr, width, height, real_range, imag_range, centered_at,cmap, samples)
    #im, p = pentadiagonal(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples)

    print(f"Saving image...{width}_{height}_samples_{samples}")


    name = f"img_{width}_{height}_samples_{samples}_{random.random()}.png"

    if width >= 3000:
        name = 'morethan4000/' + name
    else:
         name = 'examples/' + name

    im.save(name)


if __name__=='__main__':
    run_big_picture()
    #print(generate_one_comp_m(5, [1,1]))
    