import numpy as np
import random 

import math
import cmath
import itertools

from multiprocessing import Pool, TimeoutError


from numpy.linalg import LinAlgError
from tqdm import tqdm

from PIL import Image
from matplotlib import cm

import matplotlib.pyplot as plt

from numba import jit

from datetime import datetime

from scipy.linalg import circulant


values_for_animation = np.array([complex(0, 1), complex(0, 1)])

def generate_one_bohem_m_beta(n, elemets):

    A = (2)*np.random.beta(0.01, 0.01, (n, n))  - 1
    #A = A + (2j)*np.random.beta(0.01, 0.01, (n, n))   -  1j

    return A


def generate_one_bohem_m(n, elemets):

    A = np.random.choice(elemets, (n,n))


    return A

def generate_symmetric_m_beta(n, elemets):

    # A = (2)*np.random.beta(0.01, 0.01, (n, n))  - 1
    # A = A + (2j)*np.random.beta(0.01, 0.01, (n, n))   -  1j

    A = np.random.choice(elemets, (n,n))

    A = np.tril(A) + np.transpose(np.tril(A))

    np.fill_diagonal(A, 1)

    return A

def generate_one_circulant_m(n, elemets):

    p = np.random.beta(0.01, 0.01, (n))  - 1
    p[2] = 5
    p[0] = 5

    #p = p + (2j)*np.random.beta(0.01, 0.01, (n))   -  1j

    #p = np.random.choice(elemets, (n))
    #p = random_in_5gon_arr_beta(0.01, 0.01, (n))

    # p = np.random.dirichlet((0.01, 0.01, 0.01), (n) )  
    # p = p[:, 0] + p[:, 1]*1j  - 2 - 2j

    return  circulant(p) #np.random.permutation(np.diag(p)) 

def generate_quater_block_m_beta(n, elemets, generator = generate_one_circulant_m):

    MA1 = generator(n, elemets)
    MA2 = generator(n, elemets)

    MA1b = np.conjugate(MA1)
    MA2bm = (-1)*np.conjugate(MA2)

    M = np.block([[MA1, MA2],
                  [MA2bm, MA1b]])    

    return M    


@jit
def log_density_map(val, max_count): 

    brightness = math.log(val) / math.log(max_count)
    gamma = 3.2 #7.2
    brightness = math.pow(brightness, 1/gamma)

    return brightness

t = 1

# Converts a point on the complex plane to an integer 
# (x, y) coordinate within an image w/ a width and height


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



    
def random_circulant_matrix(batch_size, n, values):
    A = np.zeros((batch_size, n, n)).astype(complex)
    
    for i in range(batch_size):
        #p = (2)*np.random.beta(0.01, 0.01, (n))  - 1
        
        #p = np.random.choice(values, (n,n))

        #p = (1)*np.random.beta(0.01, 0.01, (n))  + 2
        #p = p + (1j)*np.random.beta(0.01, 0.01, (n))   + 2j

        #p = random_in_5gon_arr_beta(0.01, 0.01, (n)) 
        #p = random_in_quaddiag_arr_beta(0.01, 0.01,(n)) 

        # p = np.random.dirichlet((0.01, 0.01, 0.01), (n) )  
        # p = p[:, 0] + p[:, 1]*1j + 2 + 2j

        

        p = random_in_7gon_arr_beta(0.01, 0.01,(n)) 

        p[2] = 5
        # p[4] = 5 
        A[i, :, :] = circulant(p)

        #A[i, :, :] =n p.random.permutation(np.diag(p))
    warn = 0
    try:
        E = np.linalg.eigvals(A)
    except LinAlgError:
        E = 0
        warn = 1
        

    return E, warn

def random_permut_matrix(batch_size, n, values):
    A = np.zeros((batch_size, n, n)).astype(complex)
    
    for i in range(batch_size):
        #p = (2)*np.random.beta(0.01, 0.01, (n))  - 1
        #p = np.random.choice(values, (n,n))

        # p = (2)*np.random.beta(0.01, 0.01, (n//2))  - 1
        # p = p + (2j)*np.random.beta(0.01, 0.01, (n//2))   -  1j

        
        # if n % 2 == 0:
        #     p = (2)*np.random.beta(0.01, 0.01, (n//2))  + 2
        #     p = p + (2j)*np.random.beta(0.01, 0.01, (n//2))   +  2j
        #     p = np.concatenate((p, p[::-1]), axis=None)

        # else:
        #     p = (2)*np.random.beta(0.01, 0.01, (1 + n//2))  +  2j
        #     p = p + (2j)*np.random.beta(0.01, 0.01, (1 + n//2))   +  2j
        #     p = np.concatenate((p[0:-1:], p[::-1]), axis=None)

        #p = random_in_5gon_arr_beta(0.01, 0.01, (n)) 
        #p = random_in_quaddiag_arr_beta(0.01, 0.01,(n)) 

        #p = np.random.dirichlet((0.01, 0.01, 0.01), (n) )  
        #p = p[:, 0] + p[:, 1]*1j + 2 + 2j

        #p = random_in_quaddiag_arr_beta(0.01,0.01, (n), r = 1, h = (0, 0) )

        #A[i, :, :] = circulant(p)


        #np.fill_diagonal(A[i, :, :], 1)

        #A[i, :, :] = np.random.permutation(np.diag(p))

        A[i, :, :] = generate_symmetric_m_beta(n)

    warn = 0

    try:
        E = np.linalg.eigvals(A)
    except LinAlgError:
        E = 0
        warn = 1
        

    return E, warn

def random_block_quatern_matrix(batch_size, n, values):
    n = (n//2)*2
    A = np.zeros((batch_size, n, n)).astype(complex)
    
    for i in range(batch_size):

        A[i, :, :] = generate_quater_block_m_beta(n//2, values)

    warn = 0

    try:
        E = np.linalg.eigvals(A)
    except LinAlgError:
        E = 0
        warn = 1
        

    return E, warn

def random_circulant_matrix_double(batch_size, n, n2, values):
    A = np.zeros((batch_size, n, n)).astype(complex)
    
    for i in range(batch_size):
        p = (2)*np.random.beta(0.01, 0.01, (n))  - 1 + 10 + 10j
        #p = np.random.choice(values, (n,n))

        #p = (1)*np.random.beta(0.01, 0.01, (n))  + 2
        #p = p + (1j)*np.random.beta(0.01, 0.01, (n))   + 2j

        #p = random_in_5gon_arr_beta(0.01, 0.01, (n)) 
        #p = random_in_quaddiag_arr_beta(0.01, 0.01,(n)) 

        # p = np.random.dirichlet((0.01, 0.01, 0.01), (n) )  
        # p = p[:, 0] + p[:, 1]*1j + 2 + 2j

        A[i, :, :] = circulant(p)

        
    warn = 0

    E = np.linalg.eigvals(A).flatten()
    A2 = np.zeros((batch_size, n2, n2)).astype(complex)
    

    for i in range(batch_size):
        p = np.random.choice(E, (n2)) + 10 + 10j
        A2[i, :, :] = circulant(p)

    warn = 0
    
    E2 = np.linalg.eigvals(A2)
    
    return E2, warn


# Plot of eigenvalues of random 10x10 tridiagonal matrices
def tridiagonal(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n = 10 # n x n matrix
    mono_coloring = False

    # Array for tracking # of eigvals for each pixel
    counts = np.zeros((width, height), dtype=np.uint64)

    # Non-zero elements of the matrix are sampled from this sequence
    values = np.array([complex(0, 0), complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1), 
                    complex(0.01, 0), complex(-0.01, 0), complex(0, 0.01), complex(0, -0.01)])

    # Complute eigenvalues of samples number of matrices 
    print("Computing eigenvalues...")
    for _ in tqdm(range(samples)):
        # Entries not in the sub-, super-, or main-diagonals are zero
        mat = np.full((n, n), complex(0, 0)) 
        # Set entries in sub- and super-diagonal
        for i in range(1, n): 
            mat[i, i-1] = random.choice(values)
            mat[i-1, i] = random.choice(values)
        # Set entries along main diagonal
        for i in range(n):
            mat[i, i] = random.choice(values)
        
        # Get eigvenvalues of the matrix
        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            print("Eigenvalue error")


        # Convert each eigvenvalue to a point on image
        for z in eigvals:
            x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
            # Check if it's in image
            if (0 <= x < width) and (0 <= y < height):
                counts[y, x] += 1
    print("Done!")

    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x] != 0:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else: 
                    rgba = cmap( log_density_map(counts[y, x], max_count) )
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])

    im = Image.fromarray(im_arr)
    return im


def bohem_circ_exp(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n = 3


    counts = np.zeros((width, height), dtype=np.uint64)

    values = [-1,  0, 1/1000,  1j /1000 ]
    pr = [1] + [0]*10 + [-1]
    #values = np.roots([1,0,0,0,0,-1])
    batch = 5000

    for _ in tqdm(range(samples//batch)):

        eigvalsArr, warn = random_circulant_matrix(batch, n, values)
        #eigvalsArr, warn = random_block_quatern_matrix(batch, n, values)
        
        if warn:
            continue

        for eigvals in eigvalsArr:

            #char_poly = np.poly(np.nan_to_num(eigvals, nan=0.0, posinf=10**12, neginf=-10**12)).astype(complex)
            #char_poly[-1] = 0
            #eigvals = np.roots(char_poly)
            #eigvals = np.roots(np.polyder(char_poly))

            for z in eigvals:
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)

                if (0 <= x < width) and (0 <= y < height):
                    counts[y, x] += 1

    bac = (239, 235, 216)  
    #bac = (20, 30, 20)        
    max_count = np.max(counts)

    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x]:
                rgba = cmap( log_density_map(counts[y, x], max_count) )
                im_arr[y, x, 0] = int(255 * rgba[0])
                im_arr[y, x, 1] = int(255 * rgba[1])
                im_arr[y, x, 2] = int(255 * rgba[2])
            else:
                
                im_arr[y, x, 0] =  bac[0]
                im_arr[y, x, 1] =  bac[1]
                im_arr[y, x, 2] =  bac[2]

    im = Image.fromarray(im_arr)

    #print("\nDone!2")

    return im


def pentadiagonal(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n = 20 # n x n matrix
    mono_coloring = False

    # Array for tracking # of eigvals for each pixel
    counts = np.zeros((width, height), dtype=np.uint64)

    # Non-zero elements of the matrix are sampled from this sequence
    values = np.array([ complex(0, 0),
                        complex(1, 0), complex(-1, 0), 
                        complex(0, 1), complex(0, -1),
                        complex(0.01, 0), complex(-0.01, 0),
                        complex(0, 0.01), complex(0, -0.01)])

    values = np.array([ complex(0, 0),
                        complex(0, 0.1), complex(0, -0.1),
                        complex(0.1, 0.1), complex(-0.1, 0.1),
                        complex(1, 0), complex(-1, 0)])


    par = [1, 1, 0, 1, 0]  #+1 -1 +2 -2 0

    for i in range(5):
        par[i]  = random.choice([1, 0])

    par = [1, 0, 1, 0, 1]
    

    print("Computing eigenvalues...")
    for _ in tqdm(range(samples)):
        # Entries not in the sub-, super-, or main-diagonals are zero
        mat = np.full((n, n), complex(0,0)) 
            # Set entries in sub- and super-diagonal

        for i in range(2, n): 
            mat[i, i-2] = random.choice(values) if par[0] else np.random.beta(0.1, 0.1)*2 -1
            mat[i-2, i] = random.choice(values) if par[1] else np.random.beta(0.1, 0.1)*2 -1

        # for i in range(2, n): 
        #     mat[i, i-2] = random.choice(values) if par[2] else random_inside_square()
        #     mat[i-2, i] = random.choice(values) if par[3] else random_in_circle()
            
        # for i in range(n):
        #     mat[i, i]  =  random.choice(values)  if par[4] else random_inside_square()
            
        mat[0, 0], mat[n-1, n-1] = 1,  1

        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            pass
            #print("Eigenvalue error")


        # Convert each eigvenvalue to a point on image
        for z in eigvals:
            x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
            # Check if it's in image
            if (0 <= x < width) and (0 <= y < height):
                counts[y, x] += 1
    print("Done!")

    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
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

    par  = par + [list(values)]
    im = Image.fromarray(im_arr)
    return im, par


def circulant_image(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n = 20 # n x n matrix
    mono_coloring = 1

    # Array for tracking # of eigvals for each pixel
    counts = np.zeros((width, height), dtype=np.uint64)

    # Non-zero elements of the matrix are sampled from this sequence
    values = np.array([ complex(1, 0), complex(-1, 0), 
                        complex(0, 1), complex(0, -1)])

    # Complute eigenvalues of samples number of matrices 
    print("Computing eigenvalues...")
    for _ in tqdm(range(samples)):
        # Entries not in the sub-, super-, or main-diagonals are zero
        mat = np.full((n, n), complex(0, 0)) 
        # Set entries in sub- and super-diagonal
        
        arr_to_fill = np.array([random.choice(values) for i in range(n)])
        #arr_to_fill[19] =random_in_square()

        #arr_to_fill[18] =random_in_square()
        #arr_to_fill[19] =random.random()*2 -1
        #arr_to_fill[19] = np.random.beta(1, 1)*2 -1
        arr_to_fill[19] = random_in_circle(1 , (-1, -1))
        #arr_to_fill[2] = random_in_circle()
        arr_to_fill[14] = random_in_circle(1 , (1, 1))
        mat = circulant(arr_to_fill)

        
        #mat [1 , 1 ] = random_in_circle()
        #mat [19 , 19 ] = random.choice(values)

        # Get eigvenvalues of the matrix
        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            print("Eigenvalue error")


        # Convert each eigvenvalue to a point on image
        for z in eigvals:
            x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
            # Check if it's in image
            if (0 <= x < width) and (0 <= y < height):
                counts[y, x] += 1
    print("Done!")

    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x] != 0:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else: 
                    rgba = cmap( log_density_map(counts[y, x], max_count) )
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])

    im = Image.fromarray(im_arr)
    return im


# Returns image of "eigenfish", plot of eigenvalues of matrices w/ entries in
# {-1, 0, 1}, and some entries taking on random continous values in a specified range
def eigenfish(im_arr, width, height, real_range, imag_range, centered_at, samples):
    # A and B are at (2, 2) and (3, 3) respectively, and take on values
    # in (3, 3)
    counts = np.zeros((width, height), dtype=np.uint64)
    mono_coloring  = 0
    mat = np.array([[0, 0, -1, 0],
                    [1, -1, 0, 1], 
                    [0, 0, 0, 1],
                    [1, 1, -1, 0]], dtype=float)

    for n in tqdm(range(samples)):
        # Set A and B to random number in (-3, 3)
        A = random.uniform(-3, 3)
        B = random.uniform(-3, 3)
        mat[2, 2] = A
        mat[3, 3] = B

        # Get eigvenvalues of the matrix
        eigvals = np.linalg.eigvals(mat)

        # Convert each eigvenvalue to a point on image
        for z in eigvals:
            if z.imag == 0:
                continue
            x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
            # Check if it's in image
            if (0 <= x < width) and (0 <= y < height):
                counts[y, x] += 1

    max_count = np.max(counts)

    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x] != 0:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else: 
                    rgba = cmap( log_density_map(counts[y, x], max_count) )
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])

    im = Image.fromarray(im_arr)
    return im



# Plot of eigenvalues of companion matrices of littlewood polynomials. 
# Coefficients take on values in {-1, 1}
def littlewood(im_arr, width, height, real_range, imag_range, centered_at, cmap, input_file=None, degree=14):
    mono_coloring = False
    print("Rendering Littlewood fractal...")
    # Array for tracking # of eigvals for each pixel
    counts = np.zeros((width, height), dtype=np.uint64)

    # Read roots from input file instead of computing them
    if input_file is not None:
        print("Reading roots from input file...")
        with open(input_file) as file:
            for line in tqdm(file):
                line = line.split(" ")
                z = complex(float(line[0]), float(line[1]))
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
                if (0 <= x < width) and (0 <= y < height):
                    counts[y, x] += 1
    else: 
        coefficients = np.zeros((degree), dtype=np.int16) # Coefficients are all initialized to -1

        # Initialize the companion matrix
        companion = np.zeros((degree-1, degree-1), dtype=float) 
        for i in range(1, degree-1):
            companion[i, i-1] = 1 # Set sub-diagonal to 1's
        
        # Iterate through every permutation of coefficients and
        # compute the eigenvalues of the companion matrix    
        print("Computing eigenvalues...")
        for n in tqdm(range(pow(2, degree))):
            for i in range(degree):
                # Essentially we are counting from binary, so we retrieve each bit of the
                # current iteration number and map it to -1 or 1
                coefficients[i] = -1 if (1 & (n >> i) == 1) else 1

            if coefficients[degree-1] == 1:
                monic = True
            else:
                monic = False

            # Construct companion matrix from polynomial
            # Final column is (-a_0, -a_1, ..., -a_(n-1))
            # If polynomial is not monic, invert every coefficient to make it monic
            for i in range(degree-1):
                a = coefficients[i]
                a = -a if not monic else a
                companion[i, -1] = -a

            # Get eigenvalues of the copmpanion matrix
            eigvals = np.linalg.eigvals(companion)

            # Convert each eigvenvalue to a point on image
            for z in eigvals:
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
                # Check if it's in image
                if (0 <= x < width) and (0 <= y < height):
                    if z.imag != 0:
                        counts[y, x] += 1

    # Get maximum eigenvalue count in the array
    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x] != 0:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else:
                    brightness = math.log(counts[y, x]) / math.log(max_count)
                    gamma = 2.2
                    brightness = math.pow(brightness, 1/gamma)
                    rgba = cmap(brightness)
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])

    im = Image.fromarray(im_arr)
    return im


# Computes all complex roots of all Littlewood polynomials of specified degree,
# and writes them out to a file

def compute_littlewood_roots(degree, outfile):
    fd = open(outfile, "w")
    coefficients = np.zeros((degree), dtype=np.int16) # Coefficients are all initialized to -1

    # Initialize the companion matrix
    companion = np.zeros((degree-1, degree-1), dtype=float) 
    for i in range(1, degree-1):
        companion[i, i-1] = 1 # Set sub-diagonal to 1's
    
    # Iterate through every permutation of coefficients and
    # compute the eigenvalues of the companion matrix    
    print("Computing eigenvalues...")
    for n in tqdm(range(pow(2, degree))):
        for i in range(degree):
            # Essentially we are counting from binary, so we retrieve each bit of the
            # current iteration number and map it to -1 or 1
            coefficients[i] = -1 if (1 & (n >> i) == 1) else 1

        if coefficients[degree-1] == 1:
            monic = True
        else:
            monic = False

        # Construct companion matrix from polynomial
        # Final column is (-a_0, -a_1, ..., -a_(n-1))
        # If polynomial is not monic, invert every coefficient to make it monic
        for i in range(degree-1):
            a = coefficients[i]
            a = -a if not monic else a
            companion[i, -1] = -a

        # Get eigenvalues of the copmpanion matrix
        eigvals = np.linalg.eigvals(companion)

        # Write out each complex eigenvalue to the outfile in the form:
        # real imag
        for z in eigvals:
            if z.imag == 0:
                continue
            fd.write(str(z.real) + " " + str(z.imag) + "\n") 

    fd.close()


def compute_two_comp_m(degree, samples,  coefficients = [-1 , 1]):
    #http://www.bohemianmatrices.com/gallery/DoublyCompanion_19x19_1/
    n = degree # n x n matrix
    mono_coloring = False

    arr_of_eigs = []

    values = coefficients

    
    for q in tqdm(range(samples)):

        mat = np.full((n, n), complex(0, 0)) 

        for i in range(0, n): 
            mat[0, i]   = random.choice(values)
            mat[i, n-1] = random.choice(values)

        for i in range(0, n-1): 
            mat[i + 1, i] = 0
            mat[i, i + 1] = 0

        #try this
        # for i in range(0, n-1):
        #    mat[i+1, i] = random.choice(values)

        #for i in range(0, n):
        #    mat[i, i] =  random.choice(values)

        for i in range(n):
            for j in range(n):
                mat[i, j] = random.choice(values)

        #mat[0, 0], mat[n-1, n-1] = 100, 100

        #mat[n-1, 0], mat[0, n-1] = 100, 100

        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            print("Eigenvalue error")

        for z in eigvals:
           arr_of_eigs.append(z)
    
    return arr_of_eigs


def comp_bohem_with_const(degree, samples,  coefficients = [ -1j ,  1]):
    #http://www.bohemianmatrices.com/gallery/DoublyCompanion_19x19_1/
    n = degree # n x n matrix


    arr_of_eigs = []

    values = coefficients

    values = np.array([complex(1, 0), complex(-1, 0)])
                    #complex(0.01, 0), complex(-0.01, 0), complex(0, 0.01), complex(0, -0.01)
    
    for q in tqdm(range(samples)):

        mat = np.full((n, n), complex(0, 0)) 
        
        for i in range(0, n): 
            mat[0, i]   = random.choice(values)
            mat[i, n-1] = random.choice(values)

        for i in range(0, n-1): 
            mat[i + 1, i] = 1

        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            print("Eigenvalue error")

        char_poly = np.poly(eigvals)
        char_poly[-1] = 2 * (2**(degree))
        eigvals = np.roots(char_poly)

        for z in eigvals:
           arr_of_eigs.append(z)
    
    return arr_of_eigs


def run_animation():
    # Size of complex window w/ respect to a center focal point
    centered_at = complex(0, 0)

    real_offset = (-1, 1)
    imag_offset = (-1, 1)

    real_range = real_offset[1] - real_offset[0]
    imag_range = imag_offset[1] - imag_offset[0]
    scale = 3.6

    real_range *= scale
    imag_range *= scale

    samples = 4_00_000
    aspect_ratio = real_range / imag_range
    width = 2370
    height = int(width * 1 / aspect_ratio)

    
    
    # Initialize color map
    # cmap = cm.get_cmap("jet")
    # cmap = cm.get_cmap("viridis")
    # cmap = cm.get_cmap("nipy_spectral")
    cmap = cm.get_cmap("hot")
    
    frames = 1200
    dt = 0.006
    #dt = 0.1
    #im = eigenfish(im_arr, width, height, real_range, imag_range, centered_at, samples)
    # im = littlewood(im_arr, width, height, real_range, imag_range, centered_at, cmap)

    for i in range(frames):

        values_for_animation[0] = values_for_animation[0] * complex(1, dt) / abs(complex(1, dt))
        values_for_animation[1] = values_for_animation[1] * complex(1, -dt) / abs(complex(1, -dt))

        im_arr = np.zeros((height, width, 3), dtype=np.uint8)
        im = tridiagonal_exp(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples)

        #print(f"Saving image...{width}_{height}_samples_{samples}")
        print(f"{i} done ")
        im.save(f"N_{i}_img{width}_{height}_{values_for_animation[0]}_{values_for_animation[1]}_samples_{samples}.png")
        

def run_big_picture(it):
    # Size of complex window w/ respect to a center focal point
    centered_at = complex(0, 0)

    r = 1
    
    real_offset = (centered_at.real - r, centered_at.real + r)
    imag_offset = (centered_at.imag - r, centered_at.imag + r)

    real_range = real_offset[1] - real_offset[0]
    imag_range = imag_offset[1] - imag_offset[0]

    scale = 14 #12 #3.6

    real_range *= scale
    imag_range *= scale

    samples = 5_00_000 
    
    aspect_ratio = real_range / imag_range
    width = 2999
    height = int(width * 1 / aspect_ratio)

    
    
    # Initialize color map
    # cmap = cm.get_cmap("jet")
    # cmap = cm.get_cmap("viridis")
    cmap = cm.get_cmap("copper_r")
    #cmap = cm.get_cmap("copper_r")

    
    #im = littlewood(im_arr, width, height, real_range, imag_range, centered_at, cmap)

 
    im_arr = np.zeros((height, width, 3), dtype=np.uint8)
    im = bohem_circ_exp(im_arr, width, height, real_range, imag_range, centered_at,cmap, samples)
    #im, p = pentadiagonal(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples)

    p = 0

    print(f"Saving image...{width}_{height}_samples_{samples}")

    name = f"img_{it}_{width}_{height}_samples_{samples}_{p}_{random.random()}.png"

    if width >= 3000:
        name = 'morethan4000/' + name
    else:
         name = 'examples2/' + name

    im.save(name)


def testM():
    values = [ 1]
    n = 10
    mat = np.full((n, n), 0) 
        # Set entries in sub- and super-diagonal

    for i in range(1, n): 
        mat[i, i-1] = random.choice(values)
        mat[i-1, i] = random.choice(values)

    for i in range(0, n-1):
        mat[n - i - 1, i + 1] = random.choice(values)
        mat[n - i  - 2,  i] = random.choice(values)
        # Set entries along main diagonal

    for i in range(n):
        mat[i, i] = random.choice(values)
        mat[n - i - 1, i ] = random.choice(values)
    print(mat)

        

def test_t_v(n, samples, values = [1,-1]):
    mat = np.full((n, n), complex(0, 0)) 

    for i in range(0, n): 
        mat[0, i] += random.choice(values)
        mat[i, n-1] += random.choice(values)

    #try this
    #for i in range(0, n-1):
    #    mat[i+1, i] = random.choice(values)

    for i in range(0, n-1):
        mat[i+1, i] = 1

    mat[0, 0], mat[n-1, n-1] = 100, 100

    return mat

def random_in_square(r = 1, h = (0,0)):
    s = random.choice([1,2,3,4])
    if s == 1:
        return random.uniform(-r/2, r/2) + (r/2)*1j + h[0] + h[1]*1j
    if s == 2:
        return random.uniform(-r/2, r/2) - (r/2)*1j + h[0] + h[1]*1j
    if s == 3:
        return random.uniform(-r/2, r/2)*1j + (r/2) + h[0] + h[1]*1j
    if s == 4:
        return random.uniform(-r/2, r/2)*1j - (r/2) + h[0] + h[1]*1j

def random_in_square_arr_beta(a = 0.01, b = 0.01, dim = (5), r = 1, h = (0,0) ):
    s = random.choice([1,2, 3])
    if s == 1:
        return r*np.random.beta(a, b, dim) - r/2 + (r/2)*1j + h[0] + h[1]*1j
    if s == 2:
        return r*np.random.beta(a, b, dim) - r/2 - (r/2)*1j + h[0] + h[1]*1j
    if s == 3:
        return (r*np.random.beta(a, b, dim)- r/2)*1j + (r/2) + h[0] + h[1]*1j
    if s == 4:
        return (r*np.random.beta(a, b, dim)- r/2)*1j - (r/2) + h[0] + h[1]*1j


def random_in_trinagle_arr_beta(a = 0.01, b = 0.01, dim = (5), r = 1, h = (0,0) ):
    
    t = r*np.roots([1, 0, 0, -1]) + h[0] + h[1]*1j  #- 5j
    
    A = np.random.beta(a, b, dim).astype(complex) 

    for i in range(dim):

        s = random.choice([1,2,3])

        if s == 1:
            A[i] = A[i]*t[0] + (1 - A[i])*t[1]
        if s == 2:
            A[i] = A[i]*t[1] + (1 - A[i])*t[2]
        if s == 3:
            A[i] = A[i]*t[2] + (1 - A[i])*t[0]


    return A

def random_in_quaddiag_arr_beta(a = 0.01, b = 0.01, dim = (5), r = 1, h = (-0.5,-0.5) ):
    
    t = r*np.array([0, 1, 1 + 1j, 1j]) + h[0] + h[1]*1j
    A = np.random.beta(a, b, dim).astype(complex) 

    for i in range(dim):

        s = random.choice([1,2,3,4])

        if s == 1:
            A[i] = A[i]*t[0] + (1 - A[i])*t[1]

        if s == 2:
            
            A[i] = A[i]*t[1] + (1 - A[i])*t[2]
        if s == 3:
            
            A[i] = A[i]*t[2] + (1 - A[i])*t[3]

        if s == 4:
            
            A[i] = A[i]*t[3] + (1 - A[i])*t[0]

    return A

    # if s == 5:
        
    #     A = A*t[0] + (1 - A)*t[2]

    #     return A

    # if s == 6:
        
    #     A = A*t[1] + (1 - A)*t[3]

def random_in_5gon_arr_beta(a = 0.01, b = 0.01, dim = (5), r = 1, h = (0,0) ):

    t = r*np.roots([1, 0, 0, 0, 0, -1])  + h[0] + h[1]*1j 
    

    t = np.array( sorted(list(t), key = np.angle) ) #- 8j
    A = np.random.beta(a, b, dim).astype(complex) 

    for i in range(dim):

        s = random.choice([1,2,3,4,5])

        if s == 1:
            A[i] = A[i]*t[0] + (1 - A[i])*t[1]

        if s == 2:
            
            A[i] = A[i]*t[1] + (1 - A[i])*t[2]
        if s == 3:
            
            A[i] = A[i]*t[2] + (1 - A[i])*t[3]

        if s == 4:
            
            A[i] = A[i]*t[3] + (1 - A[i])*t[4]
        if s == 5:
            
            A[i] = A[i]*t[4] + (1 - A[i])*t[0]

        
    return A


def random_in_5star_arr_beta(a = 0.01, b = 0.01, dim = (5), r = 1, h = (0,0) ):

    t = r*np.roots([1, 0, 0, 0, 0, -1])  + h[0] + h[1]*1j 
    

    t = np.array( sorted(list(t), key = np.angle) ) - 8j
    A = np.random.beta(a, b, dim).astype(complex) 

    for i in range(dim):

        s = random.choice([1,2,3,4,5])

        if s == 1:
            A[i] = A[i]*t[0] + (1 - A[i])*t[2]

        if s == 2:
            
            A[i] = A[i]*t[2] + (1 - A[i])*t[4]
        if s == 3:
            
            A[i] = A[i]*t[4] + (1 - A[i])*t[1]

        if s == 4:
            
            A[i] = A[i]*t[1] + (1 - A[i])*t[3]
        if s == 5:
            
            A[i] = A[i]*t[3] + (1 - A[i])*t[0]

        
    return A

def random_in_6gon_arr_beta(a = 0.01, b = 0.01, dim = (5), r = 1, h = (0,0) ):

    t = r*np.roots([1, 0, 0, 0, 0, 0, -1])  + h[0] + h[1]*1j 

    t = np.array( sorted(list(t), key = np.angle) ) #- 8j
    A = np.random.beta(a, b, dim).astype(complex) 

    for i in range(dim):

        s = random.choice([1,2,3,4,5,6])

        if s == 1:
            A[i] = A[i]*t[0] + (1 - A[i])*t[1]

        if s == 2:
            
            A[i] = A[i]*t[1] + (1 - A[i])*t[2]
        if s == 3:
            
            A[i] = A[i]*t[2] + (1 - A[i])*t[3]

        if s == 4:
            
            A[i] = A[i]*t[3] + (1 - A[i])*t[4]
        if s == 5:
            
            A[i] = A[i]*t[4] + (1 - A[i])*t[5]
        if s == 6:
            
            A[i] = A[i]*t[5] + (1 - A[i])*t[0]

        
    return A    

def random_in_7gon_arr_beta(a = 0.01, b = 0.01, dim = (5), r = 1, h = (0,0) ):

    t = r*np.roots([1, 0, 0, 0, 0, 0,0, -1])  + h[0] + h[1]*1j 

    t = np.array( sorted(list(t), key = np.angle) ) #- 8j
    A = np.random.beta(a, b, dim).astype(complex) 

    for i in range(dim):

        s = random.choice([1,2,3,4,5,6, 7])

        if s == 1:
            A[i] =  (A[i]*t[0] + (1 - A[i])*t[1])
            #A[i] =  A[i] + 2

        if s == 2:
            
            A[i] = A[i]*t[1] + (1 - A[i])*t[2]
        if s == 3:
            
            A[i] = A[i]*t[2] + (1 - A[i])*t[3]

        if s == 4:
            
            A[i] = A[i]*t[3] + (1 - A[i])*t[4]
        if s == 5:
            
            A[i] = A[i]*t[4] + (1 - A[i])*t[5]
        if s == 6:
            
            A[i] = A[i]*t[5] + (1 - A[i])*t[6]

        if s == 7:
            
            A[i] = A[i]*t[6] + (1 - A[i])*t[0]

        
    return A   

def random_in_8gon_arr_beta(a = 0.01, b = 0.01, dim = (5), r = 1, h = (0,0) ):

    t = r*np.roots([1, 0, 0, 0, 0, 0,0,0, -1])  + h[0] + h[1]*1j 

    t = np.array( sorted(list(t), key = np.angle) ) #- 8j
    A = np.random.beta(a, b, dim).astype(complex) 

    for i in range(dim):

        s = random.choice([1,2,3,4,5,6, 7, 8])

        if s == 1:
            A[i] = A[i]*t[0] + (1 - A[i])*t[1]

        if s == 2:
            
            A[i] = A[i]*t[1] + (1 - A[i])*t[2]
        if s == 3:
            
            A[i] = A[i]*t[2] + (1 - A[i])*t[3]

        if s == 4:
            
            A[i] = A[i]*t[3] + (1 - A[i])*t[4]
        if s == 5:
            
            A[i] = A[i]*t[4] + (1 - A[i])*t[5]
        if s == 6:
            
            A[i] = A[i]*t[5] + (1 - A[i])*t[6]

        if s == 7:
            
            A[i] = A[i]*t[6] + (1 - A[i])*t[7]

        if s == 8:
            
            A[i] = A[i]*t[7] + (1 - A[i])*t[0]

        
    return A       

def random_inside_square(r = 1, h = (0,0)):

    return random.uniform(-r/2, r/2) + random.uniform(-r/2, r/2)*1j + h[0] + h[1]*1j


def random_in_circle(r = 1, h = (0,0)):
    theta = random.uniform(-math.pi, math.pi)
    return r*math.cos(theta) + r*math.sin(theta)*1j + h[0] + h[1]*1j

def random_in_circle_arr( dim = (5), r = 1, h = (0,0)):
    #theta = np.random.uniform(0, math.pi/2, dim)
    theta = np.random.beta(0.01, 0.01,  dim) * math.pi*t
    return r*np.cos(theta) + r*np.sin(theta)*1j + h[0] + h[1]*1j

def run_inc_pow(it):

    power = it

    base = 2

    diam = round(30 - 2*it) + 3
    diam = 10
    if diam < 3:
        diam = 4

    N = 3000

    cmap = 'hot'

    scale = N*0.06 

    size = 5

    coef = np.zeros((N, N), dtype = np.int32)
    
    arr_eigs = comp_bohem_with_const(power,  1000 +  (base*base)**power)

    for j in tqdm(arr_eigs):

        if abs(np.imag( j )) > 0.000000001 and abs(np.real( j )) > 0.0000000001:
            x = round(np.imag( j ) * scale + N/2)
            y = round(np.real( j ) * scale + N/2)

            r = diam//2
            for c in range(diam+1):
                for d in range(diam+1):

                    tempx = x + c - r
                    tempy = y + d - r
                    dist = math.dist((tempx, tempy), (x, y))

                    if dist < r and tempx < N and tempx > 0 and tempy < N and tempy > 0 :
                        coef[tempx, tempy] += r/(dist + 1) #+=

            
    
    #coef = np.rot90(coef)

    #filenameArr = f'K{it}_N_{N}_p_{power}_arg1_{arg1.real}_{arg1.imag}_arg2_{arg2.real}_{arg2.imag}_arg3_{arg3.real}_{arg3.imag}'
    #np.save(filenameArr, coef)

    ####
    max_count = np.max(coef)
    print(max_count)
    for i in range(N):
        for j in range(N):
            if coef[i, j]:
                if coef[i, j] > 120:
                    coef[i, j] = 120 
                else:
                    coef[i, j] += 40
                
    ####

    plt.figure(num = None, figsize=(size, size), dpi=300)

    plt.axis('off')

    plot = plt.imshow(coef, cmap = cmap ) #, interpolation='lanczos'

    ####

    now =  str(datetime.now()).replace(" ", '_')
    now =  now.replace(":", '_')

    filenameImage = f'R_{it}_N_{N}_cmap_{cmap}_p_{power}_{now}.png'

    if size >= 10:
        filenameImage = 'morethan4000/' + filenameImage
    else:
        filenameImage = 'examples/' + filenameImage

    plt.savefig(filenameImage, bbox_inches = 'tight', pad_inches=0.0 )

    #plt.show()
    plt.close()


def grid_open(args):

    return grid_1(*args)


def grid_1(width, height, real_range, imag_range, centered_at, samples):
    n = 8
    counts = np.zeros((width, height), dtype=np.uint64)

    values = np.array([ #complex(0, 0),
                        complex(1, 0), complex(-1, 0) ])

    
    for _ in range(samples):

        mat = np.full((n, n), complex(0,0)) 
        
        for i in range(n):
            for j in range(n):
                mat[i, j] = random.choice(values)

        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            pass
            #print("Eigenvalue error")

        char_poly = np.poly(eigvals)
        char_poly = char_poly / char_poly[0]
        char_poly[-1] = 10
        eigvals = np.roots(char_poly)


        
        for z in eigvals:
            x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
            if (0 <= x < width) and (0 <= y < height):
                counts[y, x] += 1

    return counts


def parralel_bohem(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n = 8 # n x n matrix

    counts = np.zeros((width, height), dtype=np.uint64)

    mono_coloring = False
    n_proc = 10

    for x in pool.map(grid_1, [(width, height, real_range, imag_range, centered_at, samples/n_proc) for i in range(n_proc)] ):
            counts = counts + x

    print("Done!")

    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
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

    par  =  list(values)
    im = Image.fromarray(im_arr)

    return im, par


def bohem_with_const(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n = 55 # n x n matrix

    mono_coloring = False

    # Array for tracking # of eigvals for each pixel
    counts = np.zeros((width, height), dtype=np.uint64)

    #values = np.array([complex(1, 0), complex(-1, 0)])
    # values = np.array([ complex(1, 1)/abs(complex(1, 1)), complex(-1, 0), complex(0.1, 0)])
    # values = np.array([ complex(0, 0), complex(1, 0), complex(-1, 0)])

    values = np.roots([1, 0, 0, -1])

    base = len(values)
    print("Computing eigenvalues...")

    for _ in tqdm(range(samples)):
        
        mat = np.full((n, n), complex(0, 0)) 
        
        for i in range(0, n): 
            mat[0, i]   += random.choice(values)
            mat[i, n-1] += random.choice(values)

        for i in range(0, n-1): 
            mat[i + 1, i] = 1

        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            pass
            #print("Eigenvalue error")

        char_poly = np.poly(eigvals)
        char_poly[-1] = 2**(n + 23)
        eigvals = np.roots(char_poly)


        # Convert each eigvenvalue to a point on image
        for z in eigvals:
            if abs(z.imag) > 0.00000001:
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
                # Check if it's in image
                if (0 <= x < width) and (0 <= y < height) and (counts[y, x] < 48):
                    counts[y, x] += 1
    print("Done!")

    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x]:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else: 
                    #rgba = cmap( log_density_map(counts[y, x], max_count) )
                    rgba = cmap( counts[y, x]/ max_count )
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])
            # else:
            #     rgba = cmap( 0 )
            #     im_arr[y, x, 0] = int(255 * rgba[0])
            #     im_arr[y, x, 1] = int(255 * rgba[1])
            #     im_arr[y, x, 2] = int(255 * rgba[2])

    #par  =  list(values)
    im = Image.fromarray(im_arr)

    return im

def interpolation_bohem_gen(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples):
    n = 55 
    counts = np.zeros((width, height), dtype=np.uint64)
    #values = np.roots([1, 0, 0, -1])


    base = len(values)



    for par in tqdm(range(samples)):
        
        mat = np.full((n, n), complex(0, 0)) 
        
        for i in range(0, n): 
            mat[0, i]   += random.choice(values)
            mat[i, n-1] += random.choice(values)

        for i in range(0, n-1): 
            mat[i + 1, i] = 1

        try:
            eigvals = np.linalg.eigvals(mat)
        except LinAlgError:
            pass
            #print("Eigenvalue error")

        char_poly = np.poly(eigvals)
        char_poly[-1] = 2**(n + 23)
        eigvals = np.roots(char_poly)


        # Convert each eigvenvalue to a point on image
        for z in eigvals:
            if abs(z.imag) > 0.00000001:
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)
                # Check if it's in image
                if (0 <= x < width) and (0 <= y < height) and (counts[y, x] < 48):
                    counts[y, x] += 1
    print("Done!")

    max_count = np.max(counts)
    print("Max count:", max_count)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if counts[y, x]:
                if mono_coloring is True:
                    im_arr[y, x] = 255
                else: 
                    #rgba = cmap( log_density_map(counts[y, x], max_count) )
                    rgba = cmap( counts[y, x]/ max_count )
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])
            # else:
            #     rgba = cmap( 0 )
            #     im_arr[y, x, 0] = int(255 * rgba[0])
            #     im_arr[y, x, 1] = int(255 * rgba[1])
            #     im_arr[y, x, 2] = int(255 * rgba[2])

    #par  =  list(values)
    im = Image.fromarray(im_arr)

    return im



if __name__ == "__main__":
    # fr = 100000
    # dt = ((10**4)*1j ) / fr
    
    # for i in range(1, 10):
    #     run_big_picture(i)
    #     t =  t - i*(1/fr)

    # fr = 1000
    # for i in range(1, 10):
    #     run_big_picture(i)
    #     t =  t - i*(1/fr)

    # fr = 100
    # for i in range(1, 10):
    #     run_big_picture(i)
    #     t =  t - i*(1/fr)

    run_big_picture(0)

    #print(generate_symmetric_m_beta(5))

    #testM()
    # for i in range(4, 19):
    #     run_inc_pow(i)

    # for i in range(2, 17):
    #     run_inc_pow(i)

    # for i in range(30):
    #     print(random_in_square())

