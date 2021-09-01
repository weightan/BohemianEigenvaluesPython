from bohemian import *
import itertools
from tqdm import tqdm

from PIL import Image
from datetime import datetime
import numpy as np
import random 
from matplotlib import cm
from numba import jit
from functools import lru_cache

t = 0
maxr = 300

def bohem_prime(n):
    return np.random.choice([-1j, 1j], size=(n,n))

def bohem_uniform(n):
    return np.random.rand(n,n)

def bohem_beta(n):
    return np.random.beta(0.01, 0.01, size = (n,n))

def compute_egvectors(generator, n, samples):
    arr = []

    for i in tqdm(range(samples)):
        w, eigvec = np.linalg.eig( generator(n) )
        for v in eigvec:
            arr.append(v)

    return arr


def projectM(plane):
    A = np.column_stack((plane[0], plane[1]))
    At = np.transpose(A)
    P = np.matmul(A, At)

    return P


def projecton(inp, plane, projM):

    #plane = [i/np.linalg.norm(i) for i in plane]

    #M = projectM(plane)

    v = np.matmul(projM, inp)

    plane = np.transpose(np.array(plane))

    plane = np.delete(plane, 2, 0)

    #try:
    x = np.linalg.solve(plane, v[:2])


    return x
    
def generate_one_bohem_m(n, elemets):

    A = np.random.choice(elemets, (n,n))

    return A



def mat_batch( rank, iters, elemets ):
    a = [0]*(iters+1)

    #rank is only 3!!!!

    A = generate_one_bohem_m(rank, elemets)
    B = generate_one_bohem_m(rank, elemets)

    for i in range( iters+1):
        a[i] =  (i/iters)*B + (1 - i/iters)*A 

        # try:
        #     a[i] = np.linalg.inv( (i/iters)*B + (1 - i/iters)*A ) 

        # except LinAlgError:

        #     a[i] = np.zeros((rank, rank))

    _, v = np.linalg.eig(a)
    butch = np.concatenate( (v[:, 0], v[:, 1], v[:, 2]) )

    return butch

def mat_batch_beta( rank, iters, elemets ):
    
    a = 2*np.random.beta(0.01, 0.01, size=(iters, rank, rank )) - 1
    _, v = np.linalg.eig(a)
    butch = np.concatenate( (v[:, 0], v[:, 1], v[:, 2]) )

    return butch


def hist_random_eigvectors_gif(generator, frames, t):
    

    N = 2160 
    scale = N*0.32

    iters = 20000
    samples = 100

    rank = 3
    elemets = [-1, 0, 1]

    arr = []
    counts = np.zeros((N, N, frames), dtype=np.uint32)

    plane = [np.array([1, 0, 0]),
             np.array([0, 1, 0])]

    pi = math.pi

    rmatrarr = [0]*20
    planesarr = [0]*20

    for i in range(20):

        theta = (t + i)*pi/maxr

        RMZ = [[math.cos(theta), -math.sin(theta), 0],
               [math.sin(theta), math.cos(theta), 0],
               [0, 0, 1]] 

        RMY = [[math.cos(theta), 0, math.sin(theta)],
               [0, 1, 0],
               [-math.sin(theta), 0, math.cos(theta)]]

        plane = [i/np.linalg.norm(i) for i in plane]

        plane[0] = np.matmul(RMY, plane[0])

        rmatrarr[i] = projectM(plane)
        planesarr[i] = plane

    

    for i in tqdm(range(samples)):

        # eigval =  generator(random.randrange(1, base**(rank*2)),
        #                     random.randrange(1, base**(rank*2)), iters, rank)
        butch =  generator(rank, iters, elemets)

        for v in butch:
            arr.append(v)

        if len(arr) > samples // 10:
            for v in arr:

                x = v[0].real*scale 
                y = v[1].real*scale 
                z = v[2].real*scale 


                for h in range(20):
                    
                    # x = abs(v[0])*scale 
                    # y = abs(v[1])*scale 
                    # z = abs(v[2])*scale 

                    prv = projecton((x,y,z), planesarr[h], rmatrarr[h])

                    xp = round(prv[0]+ N/2)
                    yp = round(prv[1]+ N/2)

                    if (0 <= xp < N) and (0 <= yp < N) :
                        counts[yp, xp, h] += 1

            arr = []

    

    params = {  'N':N,
                'scale':scale,
                'iters':iters,
                'samples':samples,
                'rank':rank,
                'elemets':elemets,
                'plane': plane}

    return counts, params


def hist_random_eigvectors_and_save(generator):
    

    N = 2160 
    scale = N*0.32

    iters = 60000
    samples = 100

    rank = 3
    elemets = [-1, 0, 1]

    arr = []
    counts = np.zeros((N, N, N), dtype=np.uint8)

    plane = [np.array([1, 0, 0]),
             np.array([0, 1, 0])]

    pi = math.pi
    theta = t*pi/maxr

    RMZ = [[math.cos(theta), -math.sin(theta), 0],
           [math.sin(theta), math.cos(theta), 0],
           [0, 0, 1]]

    RMY = [[math.cos(theta), 0, math.sin(theta)],
           [0, 1, 0],
           [-math.sin(theta), 0, math.cos(theta)]]

    plane = [i/np.linalg.norm(i) for i in plane]

    plane[0] = np.matmul(RMY, plane[0])

    M = projectM(plane)

    for i in tqdm(range(samples)):

        # eigval =  generator(random.randrange(1, base**(rank*2)),
        #                     random.randrange(1, base**(rank*2)), iters, rank)
        butch =  generator(rank, iters, elemets)

        for v in butch:
            arr.append(v)

        if len(arr) > samples // 10:
            for v in arr:

                x = round(v[0].real*scale + N/2)
                y = round(v[1].real*scale + N/2)
                z = round(v[2].real*scale + N/2)

                # x = abs(v[0])*scale 
                # y = abs(v[1])*scale 
                # z = abs(v[2])*scale 

                # prv = projecton((x,y,z), plane, M)

                # x = round(prv[0]+ N/2)
                # y = round(prv[1]+ N/2)

                if (0 <= x < N) and (0 <= y < N) and (0 <= z < N):
                    counts[x, y, z] += 1

            arr = []

    

    params = {  'N':N,
                'scale':scale,
                'iters':iters,
                'samples':samples,
                'rank':rank,
                'elemets':elemets,
                'plane': plane}

    np.save(f"DATA{t}_{params['N']}_sampl_{params['samples']}_rank_{params['rank']}_{random.random()}" , counts)

    return counts

def hist_random_eigvectors_one(generator):
    

    N = 2160 
    scale = N*0.32

    iters = 20000
    samples = 100

    rank = 3
    elemets = [-1, 0, 1]

    arr = []
    counts = np.zeros((N, N), dtype=np.uint64)

    plane = [np.array([1, 0, 0]),
             np.array([0, 1, 0])]

    pi = math.pi
    theta = t*pi/maxr

    RMZ = [[math.cos(theta), -math.sin(theta), 0],
           [math.sin(theta), math.cos(theta), 0],
           [0, 0, 1]]

    RMY = [[math.cos(theta), 0, math.sin(theta)],
           [0, 1, 0],
           [-math.sin(theta), 0, math.cos(theta)]]

    plane = [i/np.linalg.norm(i) for i in plane]

    plane[0] = np.matmul(RMY, plane[0])

    M = projectM(plane)

    for i in tqdm(range(samples)):

        # eigval =  generator(random.randrange(1, base**(rank*2)),
        #                     random.randrange(1, base**(rank*2)), iters, rank)
        butch = generator(rank, iters, elemets)

        for v in butch:
            arr.append(v)

        if len(arr) > samples // 10:
            for v in arr:

                x = v[0].real*scale 
                y = v[1].real*scale 
                z = v[2].real*scale 

                # x = abs(v[0])*scale 
                # y = abs(v[1])*scale 
                # z = abs(v[2])*scale 

                prv = projecton((x,y,z), plane, M)

                x = round(prv[0]+ N/2)
                y = round(prv[1]+ N/2)

                if (0 <= x < N) and (0 <= y < N) :
                    counts[y, x] += 1

            arr = []

    

    params = {  'N':N,
                'scale':scale,
                'iters':iters,
                'samples':samples,
                'rank':rank,
                'elemets':elemets,
                'plane': plane}

    #np.save(f"DATA{t}_{params['N']}_sampl_{params['samples']}_rank_{params['rank']}_{random.random()}" , counts)

    return counts, params


def render_b_vec_hist_pr_but():

    counts, params = hist_random_eigvectors(mat_batch_beta, 20)
    N = params['N']
    im_arr = np.zeros((N, N, 3), dtype=np.uint8)

    max_count = np.max(counts)

    #bac = (239, 235, 216)
    bac = (20, 30, 20)

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


    name = f"R{t}_{params['N']}_sampl_{params['samples']}_rank_{params['rank']}_{random.random()}.png"

    if params['N'] >= 3000:
        name = 'morethan4000/' + name
    else:
        name = 'examples/' + name

    print(params)

    im.save(name)

def render_b_vec_hist_pr_animation(t):

    counts, params = hist_random_eigvectors_gif(mat_batch_beta, 20, t)
    cmap = cm.get_cmap("copper_r")
    N = params['N']
    
    bac = (239, 235, 216)
    #bac = (20, 30, 20)

    for i in range(20):
        max_count = np.max(counts[:,:, i])
        im_arr = np.zeros((N, N, 3), dtype=np.uint8)


        for y in tqdm(range(N)):
            for x in range(N):
                if counts[y, x, i]:
                    rgba = cmap( log_density_map(counts[y, x, i], max_count) )
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


        name = f"R{t + i}_{params['N']}_sampl_{params['samples']}_rank_{params['rank']}_{random.random()}.png"

        if params['N'] >= 3000:
            name = 'morethan4000/' + name
        else:
            name = 'examples/' + name

        #print(params)

        im.save(name)

def render_b_vec_hist_pr_one():

    counts, params = hist_random_eigvectors_one(mat_batch_beta)
    cmap = cm.get_cmap("copper_r")
    N = params['N']
    
    bac = (239, 235, 216)
    #bac = (20, 30, 20)

    
    max_count = np.max(counts)
    im_arr = np.zeros((N, N, 3), dtype=np.uint8)


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


    name = f"R{t}_{params['N']}_sampl_{params['samples']}_rank_{params['rank']}_{random.random()}.png"

    if params['N'] >= 3000:
        name = 'morethan4000/' + name
    else:
        name = 'examples/' + name

    #print(params)

    im.save(name)


def hist(generator, rank, samples, N):
    arr = []
    counts = np.zeros((N, N), dtype=np.uint64)
    scale = N*0.4

    for i in tqdm(range(samples)):
        w, eigvec = np.linalg.eig( generator(rank) )
        for v in eigvec:
            arr.append(v)

        if len(arr) > 11000:
            for v in arr:
                # x = round(abs(v[0])*scale + N/2)
                # y = round(abs(v[1])*scale + N/2)

                x = round(v[0].real*scale + N/2)
                y = round(v[1].real*scale + N/2)

                #print(x, y, abs(v[2]))

                if (0 <= x < N) and (0 <= y < N):

                    # counts[y, x, 0]  = counts[y, x, 0] * (counts[y, x, 1] - 1)/counts[y, x, 1] + v[2].real*scale/counts[y, x, 1]
                    # counts[y, x, 1] += 1

                    counts[y, x] += 1#v[2].real*scale
            arr = []

    return counts   #[:, :, 0]

def plot_egvectors(N, sampl, rank):
    vectors = compute_egvectors(bohem_beta, rank, sampl)

    scale = N*0.4
    im_arr = np.zeros((N, N, 3), dtype=np.uint8)
    counts = np.zeros((N, N), dtype=np.uint64)
    #max_count = max(vectors, key = lambda i: i[2]) [2]

    cmap = cm.get_cmap("rainbow")

    for v in vectors:

        # x = round(abs(v[0])*scale + N/2)
        # y = round(abs(v[1])*scale + N/2)

        x = round(v[0].real*scale + N/2)
        y = round(v[1].real*scale + N/2)

        #print(x, y, abs(v[2]))

        if (0 <= x < N) and (0 <= y < N):
            counts[y, x] += v[2].real * scale

       
    max_count = np.max(counts)

    bac = (239, 235, 216)

    for y in tqdm(range(N)):
        for x in range(N):
            if counts[y, x]:
                rgba = cmap( log_density_map(counts[y, x], max_count) )
                im_arr[y, x, 0] = int(255 * rgba[0])
                im_arr[y, x, 1] = int(255 * rgba[1])
                im_arr[y, x, 2] = int(255 * rgba[2])
            else:
                im_arr[y, x, 0] = bac[0]
                im_arr[y, x, 1] = bac[1]
                im_arr[y, x, 2] = bac[2]

    im = Image.fromarray(im_arr)

    return im

def render_b_vec(N = 200, sampl = 1000, rank = 3):

    im = plot_egvectors(N, sampl, rank)

    name = f"N_{N}_sampl_{sampl}_rank_{rank}_{random.random()}.png"

    if N >= 3000:
        name = 'morethan4000/' + name
    else:
        name = 'examples/' + name

    im.save(name)

def render_b_vec_hist(N = 200, sampl = 1000, rank = 3):

    counts = hist(bohem_beta, rank, sampl, N)

    im_arr = np.zeros((N, N, 3), dtype=np.uint8)

    max_count = np.max(counts)

    bac = (239, 235, 216)

    cmap = cm.get_cmap("hot")

    for y in tqdm(range(N)):
        for x in range(N):
            if counts[y, x]:
                rgba = cmap( log_density_map(counts[y, x], max_count) )
                im_arr[y, x, 0] = int(255 * rgba[0])
                im_arr[y, x, 1] = int(255 * rgba[1])
                im_arr[y, x, 2] = int(255 * rgba[2])
            else:
                im_arr[y, x, 0] = bac[0]
                im_arr[y, x, 1] = bac[1]
                im_arr[y, x, 2] = bac[2]

    im = Image.fromarray(im_arr)


    name = f"N_{N}_sampl_{sampl}_rank_{rank}_{random.random()}.png"

    if N >= 3000:
        name = 'morethan4000/' + name
    else:
        name = 'examples/' + name

    im.save(name)

if __name__ == "__main__":

    # a = [generate_one_bohem_m(3, [1, 0, -1]) for i in range(3)]
    # print(a)

    # _, v = np.linalg.eig(a)
    # butch = np.concatenate( (v[:, 0], v[:, 1], v[:, 2]) )

    # print(  butch )

    # plane = [np.array([1, 1, 0]),
    #          np.array([0, 1, 1])]

    # plane = [i/np.linalg.norm(i) for i in plane]

    # M = projectM(plane)

    # v = np.matmul(M, [-3, 2, 10])

    # plane = np.transpose(np.array(plane))

    # plane = np.delete(plane, 2, 0)

    # x = np.linalg.solve(plane, v[:2])

    # print(x)


    for j in range(41, maxr):
        t = j
        render_b_vec_hist_pr_one()
        

