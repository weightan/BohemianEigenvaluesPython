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
maxr = 200

def render_b_vec_hist_pr():

    counts, params = hist_random_eigvectors(mat_batch_beta)
    N = params['N']
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


    name = f"R{t}_{params['N']}_sampl_{params['samples']}_rank_{params['rank']}_{random.random()}.png"

    if params['N'] >= 3000:
        name = 'morethan4000/' + name
    else:
        name = 'examples/' + name

    print(params)

    im.save(name)


def 


if __name__ == "__main__":

    counts = np.load('DATA0_2160_sampl_100_rank_3_0.6511772960256741.npy')

    prv = projecton((x,y,z), plane, M)

    x = round(prv[0]+ N/2)
    y = round(prv[1]+ N/2)

    for i in range(0, maxr*2):
        t = i
        render_b_vec_hist_pr()