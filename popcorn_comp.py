import numpy as np
import random 

import math
import cmath
import itertools

from tqdm import tqdm

from PIL import Image
from matplotlib import cm


def log_density_map(val, max_count): 

    brightness = math.log(val) / math.log(max_count)
    gamma = 3.2 #7.2
    brightness = math.pow(brightness, 1/gamma)

    return brightness

def flip(t):
    return t.real*1j + t.imag

def g1(t):
    return cmath.sin(t)*1j + cmath.cos(t) 

def g2(t):
    return cmath.tan(t)

def popcorn(h):

    sc = 2
    iterationsN = 21
    a = 3
    mn = 0.05

    counts = np.zeros((h, h))

    for i in tqdm(range(h)):
        for j in range(h):

            col = 0

            
            z = complex(sc/2 - sc*i/h, sc/2 - sc*j/h)

            for k in range(iterationsN):

                x = z.real
                y = z.imag*1j

                px = x

                x = x -  mn*( g1(y + g2(a*y) )).real  -    mn*(g1(x + g2(a*x))).imag*1j
                y = y -  mn*(g1(px + g2(a*px))).real     -    mn*(g1(y + g2(a*y))).imag*1j

                z = x + flip(y)

                try:
                    x = x / abs(z)
                    y = y / abs(z)

                except Exception:
                    # x = 1000
                    # y = 1000
                    pass

                z = x + flip(y)

                #print(z)

                if abs(z) > 2:
                    print('!!!')
                    break

                float angle = 45.*3.14/180.;

                x0=(real(z)*cos(angle)-imag(z)*sin(angle)-xmin)/deltap

                y0=(ymax-real(z)*sin(angle)-imag(z)*cos(angle))/deltaq;

                counts[x0, y0] += 1

    return counts

def colorize(counts, h):

    cmap = cm.get_cmap("copper")

    im_arr = np.zeros((h, h, 3), dtype=np.uint8)

    max_count = np.max(counts)
    

    for y in tqdm(range(h)):
        for x in range(h):
            if counts[y, x] > 0:
                rgba = cmap( log_density_map(counts[y, x], max_count) )

                #rgba = [counts[y, x]/max_count for i in range(3)]

                im_arr[y, x, 0] = int(255 * rgba[0])
                im_arr[y, x, 1] = int(255 * rgba[1])
                im_arr[y, x, 2] = int(255 * rgba[2])

    return im_arr


def run():
    h = 400

    counts = popcorn(h)

    im_arr = colorize(counts, h)

    print(f"Saving image...{h}")

    name = f"img_{h}_{random.random()}.png"

    if h >= 3000:
        name = 'morethan4000/' + name
    else:
         name = 'examples2/' + name

    im = Image.fromarray(im_arr)

    im.save(name)






if __name__ == "__main__":
    run()