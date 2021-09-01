import numpy as np
import cv2

# img = cv2.imread('watch.png',0)
# print img
# img_size=img.shape
# print img_size

# img_mod = np.zeros((600, 800))

# for i in range(0,img_size[0]-30):
#     for j in range(0,img_size[1]-30):
#         kernel = img[i:i+30,j:j+30]
#         for k in range(0,30):
#             for l in range(0,30):
#                 element = kernel[k,l]
#                 rank = 0
#                 for m in range(0,30):
#                     for n in range(0,30):
#                         if(kernel[k,l]>kernel[m,n]):
#                             rank = rank + 1
#                 img_mod[i,j] = ((rank * 255 )/900)

# im = np.array(img_mod, dtype = np.uint8)
# cv2.imwrite('target.png',im)


def subsample