
import time

import numpy as np

import pyopencl as cl

from PIL import Image

import matplotlib.pyplot as plt

import random

import itertools 


w = 3000
h = w
cmap = 'twilight_r'
#cmap = 'hot'
imsized = 10

cool_p = [[0.13311172873135124, 0.14528463530575508, 0.7],
          [-0.2436671664278458, -0.08564697790358555, 0.7]]

iters = 200
scaleP = 1 #4
params = cool_p[1]

def display(mesh):

    #imsized = 10

    

    plt.figure(num = None, figsize=(imsized, imsized), dpi=300)

    plt.axis('off')

    #plot = plt.imshow(mesh, cmap = cmap, interpolation='lanczos' )
    plot = plt.imshow(mesh, cmap = cmap, interpolation='lanczos')
    ####

    filenameImage = f'test{h}_{w}_{params}_{iters}_{scaleP}_{cmap}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight',  pad_inches=0.0)

    ####

    #plt.show()
    plt.close()

def calc_fractal(q, maxiter):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    output = np.empty(q.shape, dtype = np.float32)

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    prg = cl.Program(
        ctx,
        """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
     


    __kernel void mandelbrot(__global double2 *q,
                             __global float *output,
                              ushort const maxiter,
                              double const wx,
                              double const wy)
    {
        int gid = get_global_id(0);
        double const TWO_PI  = 6.283185307179586;
        double xx = q[gid].x;
        double yy = q[gid].y;
        double newx, newy = 0;
        float s = 0;
        //output[gid] = 0;
        float const maxv = 100; //40
        float const gamma = 1/0.71; // 1/0.51

        for(int curiter = 0; curiter < maxiter; curiter++) {
            newx =  wx + (xx/sin( cos(yy) ) );
            newy =  wy + (yy/cos( sin(xx) ) );

            s+=1;

            xx = newx ;
            yy = newy ;

            if((newx > 100)||(newx > 100)) {
                break;
            }
        }

        float b = log(s)/log(maxv);
        b = pow(b, gamma) ;
        output[gid] = b;
        
         
    }
    """,
    ).build()

    prg.mandelbrot(
        queue, output.shape, None, q_opencl, output_opencl, np.uint16(maxiter), np.float64(params[0]), np.float64(params[1])
    )

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output

class Mandelbrot:
    def rand_params(self,  mx = 3):
        #mx = 0.4
        params[0] = random.uniform(-mx, mx)
        params[1] = random.uniform(-mx, mx)
        print(params)

    def draw(self,  maxiter=20):
        xmin = -scaleP
        xmax = scaleP

        ymin = -scaleP
        ymax = scaleP

        xabs = abs(xmin) + abs(xmax)
        yabs = abs(ymin) + abs(ymax)

        xx = np.arange(xmin, xmax, xabs / w)
        yy = np.arange(ymin, ymax, yabs / h) *1j

        q = np.ravel(xx + yy[:, np.newaxis])

        start_main = time.time()
        output = calc_fractal(q, maxiter)
        end_main = time.time()

        secs = end_main - start_main
        print("Main took", secs)

        self.mandel = output.reshape((h, w))

    def create_image(self):

        self.draw(iters)
        display(self.mandel)
        


if __name__ == "__main__":

    #xp = np.linspace(-1, 1, 100)
    #yp = np.linspace(-1, 1, 100)
    #pars = np.array( list(itertools.product(xp, yp)))

    for i in range(4):

        test = Mandelbrot()
        #params[0], params[1] = i
        test.rand_params()

        test.create_image()
        