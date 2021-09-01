import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import circulant
import math
import random

from tqdm import tqdm

from PIL import Image
from matplotlib import cm

from numba import jit

N = 2
C_offset = (-1, -1j)
R_offset = -1 
values_main = [-1, 1]

##################

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

##################    

def beta_real_circulant(n, elements):
    offset = R_offset
    p = (2)*np.random.beta(0.01, 0.01, (n))  + offset
    return  circulant(p) 


def choice_circulant(n, elements):  

    p = np.random.choice(elements, (n))

    return  circulant(p) 


def beta_complex_circulant(n, elements):
    offset = C_offset #(-1, -1j)

    p = (2)*np.random.beta(0.01, 0.01, (n))  + offset[0]
    p = p + (2j)*np.random.beta(0.01, 0.01, (n))  + offset[1]

    return  circulant(p) 


def generate_block_matrix(n, elements, generator):

    MA1 = generator(n, elements)
    MA2 = generator(n, elements)

    MA1b = np.conjugate(MA1)
    MA2bm = (-1)*np.conjugate(MA2)

    M = np.block([[MA1, MA2],
                  [MA2bm, MA1b]])    

    return M   



##################

def batch_block_matrices(batch_size, n, values, generator):
    n = n - ( n % 2 )

    A = np.zeros((batch_size, n, n)).astype(complex)
    
    for i in range(batch_size):

        A[i, :, :] = generate_block_matrix(n//2, values, generator)

    warn = 0

    try:
        E = np.linalg.eigvals(A)
    except LinAlgError:
        E = 0
        warn = 1
        

    return E, warn


def batch_regular_circulant_matrices(batch_size, n, values, generator):
    
    A = np.zeros((batch_size, n, n)).astype(complex)
    
    for i in range(batch_size):

        A[i, :, :] = generator(n, values)

    warn = 0

    try:
        E = np.linalg.eigvals(A)
    except LinAlgError:
        E = 0
        warn = 1
        

    return E, warn

##################

def bohem_circ_block(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples, generator):
    n = N
    batch = 5000
    values = values_main
    #generator = beta_real_circulant
    #generator = choice_circulant
    #generator = beta_complex_circulant

    counts = np.zeros((width, height), dtype=np.uint64)

    bac = (239, 235, 216)  
    #bac = (20, 30, 20)  

    for _ in tqdm(range(samples//batch)):

        eigvalsArr, warn = batch_block_matrices(batch, n, values, generator)

        if warn:
            continue

        for eigvals in eigvalsArr:

            for z in eigvals:
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)

                if (0 <= x < width) and (0 <= y < height):
                    counts[y, x] += 1

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

    return im


def bohem_circ_regular(im_arr, width, height, real_range, imag_range, centered_at, cmap, samples, generator):
    n = N
    batch = 5000
    values = values_main
    #generator = beta_real_circulant
    #generator = choice_circulant
    #generator = beta_complex_circulant

    counts = np.zeros((width, height), dtype=np.uint64)

    bac = (239, 235, 216)  
    #bac = (20, 30, 20)  

    for _ in tqdm(range(samples//batch)):

        eigvalsArr, warn = batch_regular_circulant_matrices(batch, n, values, generator)

        if warn:
            continue

        for eigvals in eigvalsArr:

            for z in eigvals:
                x, y = complex_to_image(z, width, height, real_range, imag_range, centered_at)

                if (0 <= x < width) and (0 <= y < height):
                    counts[y, x] += 1

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

    return im


def render_picture_histogram(P = 'R_real'):
    
    centered_at = complex(0, 0)
    r = 1
    scale = 10 #12 #3.6
    samples = 500_000 
    width = 2500
    cmap_name = "copper_r" # "hot"

    real_offset = (centered_at.real - r, centered_at.real + r)
    imag_offset = (centered_at.imag - r, centered_at.imag + r)

    real_range = real_offset[1] - real_offset[0]
    imag_range = imag_offset[1] - imag_offset[0]

    real_range *= scale
    imag_range *= scale

    aspect_ratio = real_range / imag_range

    height = int(width * 1 / aspect_ratio)
    
    cmap = cm.get_cmap(cmap_name)
    
    im_arr = np.zeros((height, width, 3), dtype=np.uint8)

    if P == 'B_real':
        im = bohem_circ_block(im_arr, width, height, real_range, imag_range, centered_at,cmap, samples, beta_real_circulant)
    elif P == 'B_complex':
        im = bohem_circ_block(im_arr, width, height, real_range, imag_range, centered_at,cmap, samples, beta_complex_circulant)
    elif P == 'R_real':
        im =  bohem_circ_regular(im_arr, width, height, real_range, imag_range, centered_at,cmap, samples, beta_real_circulant)
    elif P == 'R_complex':
        im =  bohem_circ_regular(im_arr, width, height, real_range, imag_range, centered_at,cmap, samples, beta_complex_circulant)

    name = f"{P}_{N}_{width}_{height}_samples_{samples}_scale_{scale}.png"

    print(name)

    im.save(name)


if __name__ == "__main__":

    for i in range(2, 15):
        N = i

        if N%2 == 0:
            render_picture_histogram(P = 'B_real')
            render_picture_histogram(P = 'B_complex')

        render_picture_histogram(P = 'R_real')
        render_picture_histogram(P = 'R_complex')

   