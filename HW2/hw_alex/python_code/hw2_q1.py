

import numpy as np
from scipy import signal


def stride_conv(arr, arr2, s, mode_input):
    return signal.convolve2d(arr, arr2[::-1, ::-1], mode=mode_input, boundary='fill', fillvalue=0)[::s, ::s]


input_im = np.array([[4, 1, 6, 1, 3], [3, 2, 7, 7, 2], [2, 5, 7, 3, 7], [1, 4, 7, 1, 3], [0, 1, 6, 4, 4]])
conv_kernel = np.array([[0.1, 0.2, 0.05], [0.05, 0.2, 0.1], [0.15, 0.1, 0.05]])


# first use the stride = 2, padding = 1
res_1 = stride_conv(input_im, conv_kernel, 2, 'same')


# use stride = 1, padding = 2
res_2 = stride_conv(input_im, conv_kernel, 1, 'full')

