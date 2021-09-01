
import time

import numpy as np

import pyopencl as cl

from PIL import Image

import matplotlib.pyplot as plt

import random

import itertools 


w = 1000
h = w
cmap = 'twilight_r'
imsized = 10

cool_p = [[0.13311172873135124, 0.14528463530575508, 0.7], [0.13311172873135124, 0.14528463530575508, 0.7]]

iters = 20
scaleP = 4 #4
params = cool_p[0]

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
        output[gid] = 0;
        float const modv = 10.0; //40
        double const a = 0.7;

        double const gamma = 1/0.51; // 1/0.51
        float const mv = 100;

        for(int curiter = 0; curiter < maxiter; curiter++) {
            newx = xx + wx - (a/(TWO_PI))*sin(TWO_PI * yy ) ;
            newy = yy + wy - (a/(TWO_PI))*sin(TWO_PI * xx ) ;

            xx = newx ;
            yy = newy ;

            //*400
            //double ds =  sqrt((q[gid].x - newx)*(q[gid].x - newx) + (q[gid].y - newy)*(q[gid].y - newy))*200;

            double ds =  sqrt(fabs(newx)+fabs(newy)+20); // +20

            if(!isinf(ds)) {
                s += ds;
            }
        }

        //double b = log(fmod(s , modv))/log(mv) ;
        double b = fmod(s , modv);

        if (isnan(b) || isinf(b)){
            output[gid] = 0;
        }else{
            output[gid] = pow(b, gamma) ;
            //output[gid] = s;
        }
        
         
    }
    """,
    ).build()

    prg.mandelbrot(
        queue, output.shape, None, q_opencl, output_opencl, np.uint16(maxiter), np.float64(params[0]), np.float64(params[1])
    )

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output

class Mandelbrot:
    def rand_params(self,  mx = 0.1):
        #mx = 0.4
        params[0] = random.uniform(0, mx)
        params[1] = random.uniform(0, mx)
        print(params)

    def draw(self,  maxiter=20):

        # draw the Mandelbrot set, from numpy example
        xx = np.arange(0, scaleP, scaleP / w)
        yy = np.arange(0, scaleP, scaleP / h) *1j

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
        