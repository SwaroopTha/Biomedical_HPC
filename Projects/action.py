import numpy as np
from scipy.signal import convolve as scipy_convolve
import math
from matplotlib import pyplot as plt
import nibabel as nib
from skimage import color
from scipy.ndimage import zoom


def convolve3D(image, kernel, padding=0):
    '''
    convolve3D takes in the image array and the kernel array and convolves them
    with the padding scheme specified by parameter "padding"

    Input:
    image (Numpy.Array): 3-D image to be convolved
    kernel (Numpy.Array): 3-D kernel to be convolved
    padding (int): padding mode. default is 0

    Output:
    result (Numpy.Array): result of the convolution between image and kernel
    '''
    kernel = np.flip(kernel)

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    zKernShape = kernel.shape[2]
    padding = (math.floor(xKernShape / 2), math.floor(yKernShape / 2), math.floor(zKernShape / 2))
    
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    zImgShape = image.shape[2]

    # Shape of Output Convolution
    if xKernShape % 2 == 0:
        xOutput = int((xImgShape - xKernShape + 2 * padding[0]))
    else:
        xOutput = int((xImgShape - xKernShape + 2 * padding[0]) + 1)
        
    if yKernShape % 2 == 0:
        yOutput = int((yImgShape - yKernShape + 2 * padding[1]))
    else:
        yOutput = int((yImgShape - yKernShape + 2 * padding[1]) + 1)
    
    if zKernShape % 2 == 0:
        zOutput = int((zImgShape - zKernShape + 2 * padding[2]))
    else:
        zOutput = int((zImgShape - zKernShape + 2 * padding[2]) + 1)

    output = np.zeros((xOutput, yOutput, zOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding[0]*2, image.shape[1] + padding[1]*2, image.shape[2] + padding[2]*2))
        imagePadded[int(padding[0]):int(-1 * padding[0]), int(padding[1]):int(-1 * padding[1]), int(padding[2]):int(-1 * padding[2])] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for z in range(image.shape[2]):
        if z > image.shape[2] - zKernShape + 2*padding[2]:
            break
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yKernShape + 2*padding[1]:
                break
            # Only Convolve if y has gone down by the specified Strides

            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape + 2*padding[0]:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    output[x, y, z] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape, z: z + zKernShape]).sum()
                except:
                    break

    return output


def richardson_lucy_3d(image, psf, padding, num_iter=30, eps=1e-12):
    """Richardson-Lucy deconvolution.
    Parameters
    ----------
    image : ndarray
      Input degraded image (can be N dimensional).
    psf : ndarray
      The point spread function.
    eps: float, optional
      Value below which intermediate results become 0 to avoid division
      by small numbers.
    padding: int
      Value depends on size of psf used. Use the padding size in the "same" padding mode.
    num_iter : int, optional
      Number of iterations. This parameter plays the role of
      regularisation.
    Returns
    -------
    im_deconv : ndarray
      The deconvolved image.
    """
    #assert(math.floor(psf.sum()) == 1)
    psf_t = np.transpose(psf)
    output = np.ones(image.shape)
    
    for i in range(num_iter):
        conv = convolve3D(output, psf, padding=0) + eps
        rel_blur = image / conv
        output = output * convolve3D(rel_blur, psf_t, padding=0)
    
    return output


# action items
psf = np.ones((5, 5, 5)) / 125

noisy_by_channel = np.load("noisy_by_channel.npy")
li = []

for i in noisy_by_channel:
    deconvolved_result = richardson_lucy_3d(i, psf, num_iter=20, padding=0)
    li.append(deconvolved_result)
    
np.save("results3.npy", li)