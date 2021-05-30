#! /usr/bin/env python

''' Polynomial long division

From http://stackoverflow.com/questions/26173058/division-of-polynomials-in-python

A polynomial is represented by a list of its coefficients, eg
5*x**3 + 4*x**2 + 1 -> [1, 0, 4, 5]

Modified by PM 2Ring 2014.10.03
'''
import quaternion 
import numpy as np
import random

def normalize(poly):
    while poly and poly[-1] == np.quaternion(0,0,0,0):
        poly.pop()
    if poly == []:
        poly.append(0)


def poly_divmod(num, den):
    #Create normalized copies of the args

    num = num[:]
    normalize(num)
    den = den[:]
    normalize(den)

    if len(num) >= len(den):
        #Shift den towards right so it's the same degree as num
        shiftlen = len(num) - len(den)

        den = [np.quaternion(0,0,0,0)] * shiftlen + den
    else:
        return [0], num

    quot = []
    divisor = den[-1]

    for i in range(shiftlen + 1):
        #Get the next coefficient of the quotient.
        mult = num[-1] / divisor
        quot = [mult] + quot

        #Subtract mult * den from num, but don't bother if mult == 0
        #Note that when i==0, mult!=0; so quot is automatically normalized.
        if mult != np.quaternion(0,0,0,0):
            d = [mult * u for u in den]
            num = [u - v for u, v in zip(num, d)]

        num.pop()
        den.pop(0)

    normalize(num)
    return quot, num


def test(num, den):
    #print "%s / %s ->" % (num, den)
    q, r = poly_divmod(num, den)
    print ("quot: %s, rem: %s\n" % (q, r))
    return q, r


def main():
    num = [np.quaternion(2, 1, random.randrange(2),0) for i in range(10) ]
    #den = [np.quaternion(1, 1, random.randrange(10),0) for i in range(3) ]
    den = [np.quaternion(1,0,0,0), np.quaternion(-1,0,0,0), np.quaternion(25,0,0,0) ]

    quot, rem = poly_divmod(num, den)

    quot = [list(quaternion.as_float_array(quot[item ])) for item in range(len(quot))]
    rem = [list(quaternion.as_float_array(rem[item ] )) for item in range(len(rem))]

    quot = [[ round(quot[i][j], 2) for j in range(4)] for i in range(len(quot))]
    rem = [[ round(rem[i][j], 2) for j in range(4)]  for i in range(len(rem))]

    print("quot ", quot, '\n\n' )
    print("rem ",rem , '\n\n')


   



if __name__ == '__main__':
    main()