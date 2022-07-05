import cv2
import numpy as np
import glob
import os
import random
import math


## ========= MTB Alignment ========= ##
def MTB(img_lst, mode='rgb', plot=False):
    (h, w) = img_lst[0].shape[:2]
    if mode == 'raw':
        h = h // 2
        w = w // 2

    # Generate MTB images
    print("Generating MTB images ...")
    mtb_lst = []
    for img in img_lst:
        if mode == 'rgb':
            y = img.copy() / 64  # To avoid overflowing
            y = y.astype('uint16')
            y = (54 * y[:, :, 0] + 183 * y[:, :, 1] + 19 * y[:, :, 2]) / 256
        elif mode == 'raw':
            # Use 1 of the green pixel for generating MTB maps
            y = img[0::2, 1::2].copy()

        median = np.median(y)
        mtb = np.zeros(y.shape).astype('bool')
        for i in range(h):
            for j in range(w):
                mtb[i, j] = True if y[i, j] > median else False
        mtb_lst.append(mtb)
    
    # Plot the MTB images
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 10))
        for i in range(len(img_lst)):
            plt.subplot(3, 4, i + 1)
            plt.imshow(mtb_lst[i], cmap='gray')
            plt.axis('off')
        plt.show()

    # Calculate offsets
    print("Calculating offset ...")
    offset_lst = [[0, 0]]

    center = np.array(mtb_lst[0].shape) // 2
    if mode == 'rgb':
        patch_size = 1408
    elif mode == 'raw':
        patch_size = 704

    for index in range(1, len(img_lst)):
        step = 64

        offset = [0, 0]
        while step >= 1:
            error = np.zeros((5, 5))
            mtb_0 = mtb_lst[0][center[0] - patch_size:center[0] + patch_size:step, 
                              center[1] - patch_size:center[1] + patch_size:step]
            for i in range(-2, 3):
                for j in range(-2, 3):
                    mtb_1 = mtb_lst[index][center[0] + offset[0] + i * step - patch_size:center[0] + offset[0] + i * step + patch_size:step, 
                                           center[1] + offset[1] + j * step - patch_size:center[1] + offset[1] + j * step + patch_size:step]

                    error[i + 2, j + 2] = np.sum(np.logical_xor(mtb_0, mtb_1)) 
              
            
            offset[0] += step * (np.argmin(error) // 5 - 2)
            offset[1] += step * (np.argmin(error) % 5 - 2)
            step = step // 2
        
        if mode == 'raw':
            offset[0] *= 2
            offset[1] *= 2

        offset_lst.append(offset)
        print('Offset for image ' + str(index) + ' aligning to image 0: ' + str(offset))
    return np.asarray(offset_lst)


## ========= HDR Estimation ========= ##
LAMBDA = 5
Z_MAX = 255
Z_MIN = 0
Z_MEDIAN = (Z_MAX + Z_MIN + 1) // 2

def debevec_weight(x):
    if x > Z_MEDIAN:
        return x - Z_MEDIAN
    else:
        return Z_MEDIAN - x


def gaussian_weight(x):
    SIGMA = 0.3
    weight = math.exp(-pow((x/Z_MAX) - 0.5, 2) / (2 * pow(SIGMA, 2)))
    return weight


def de_vene_weight(x):
    a = 100
    b = 0.04
    weight = math.exp(-pow(x - Z_MEDIAN, a) / (b * pow(Z_MEDIAN, a)))
    return weight


def jpg_HDR(img_lst, offset_lst, expo_time_ln, weight_func='Gaussian'):
    # sampling pixels
    sample_idx = sorted(random.sample(range(HEIGHT*WIDTH), SAMPLE_CNT))
    sample_coord = np.zeros((SAMPLE_CNT, 2), dtype=int)  # convert to 2D coordinate which is easier to add MTB offsets
    for i in range(SAMPLE_CNT):
        sample_coord[i, 0] = sample_idx[i] // WIDTH
        sample_coord[i, 1] = sample_idx[i] % WIDTH

    # calculate irradiance
    n = 256
    irradiance = np.zeros((HEIGHT, WIDTH, 3))

    for color in range(3):
        # Check legal point count
        valid_idx = []
        for i in range(SAMPLE_CNT):
            if sample_coord[i, 0] + np.max(offset_lst[:NUM_PHOTO_USED, 0]) < HEIGHT and sample_coord[i, 0] + np.min(offset_lst[:NUM_PHOTO_USED, 0]) >= 0:
                if sample_coord[i, 1] + np.max(offset_lst[:NUM_PHOTO_USED, 1]) < WIDTH and sample_coord[i, 1] + np.min(offset_lst[:NUM_PHOTO_USED, 1]) >= 0:
                    valid_idx.append(i)
  
        # Build A, b matrices     
        N = len(valid_idx)
        A = np.zeros((N * NUM_PHOTO_USED + n + 1, N + n))
        b = np.zeros((A.shape[0], 1))

        k = 0
        for i in range(N):
            coord = sample_coord[valid_idx[i], :]
            for j in range(NUM_PHOTO_USED):
                pixel_value = img_lst[j][coord[0] + offset_lst[j, 0], coord[1] + offset_lst[j, 1], color] + 1
                if weight_func == 'Debevec':
                    w = debevec_weight(pixel_value)
                elif weight_func == 'Devene':
                    w = de_vene_weight(pixel_value)
                else:
                    w = gaussian_weight(pixel_value)
                A[k, pixel_value] = w
                A[k, n + i] = -w
                b[k] = w * expo_time_ln[j]
                k += 1
        A[k, 128] = 1
        k += 1

        for i in range(n - 2):
            if weight_func == 'Debevec':
                w = debevec_weight(i + 1)
            elif weight_func == 'Devene':
                w = de_vene_weight(i + 1)
            else:
                w = gaussian_weight(i + 1)
            A[k][i] = LAMBDA * w
            A[k][i+1] = -2 * LAMBDA * w
            A[k][i+2] = LAMBDA * w
            k += 1

        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        for i in range(HEIGHT):
            for j in range(WIDTH):
                numerator = 0
                denominator = 0
                for p in range(NUM_PHOTO_USED):
                    h_idx = i + offset_lst[p, 0]
                    w_idx = j + offset_lst[p, 1]
                    if h_idx >= 0 and h_idx < HEIGHT and w_idx >= 0 and w_idx < WIDTH:
                        curr_pixel = img_lst[p][h_idx, w_idx, color]
                        if weight_func == 'Debevec':
                            weight = debevec_weight(curr_pixel)
                        elif weight_func == 'Devene':
                            weight = de_vene_weight(curr_pixel)
                        else:
                            weight = gaussian_weight(curr_pixel)
                        numerator += weight * (x[curr_pixel][0] - expo_time_lst[p])
                        denominator += weight

                irradiance[i, j, color] = np.exp(numerator / denominator)

    return irradiance


## ========= Main Function ========= ##
if __name__ == "__main__":

    foldername = 'data/jpg/'
    expo_time_lst = [8, 4, 2, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
    expo_time_ln = np.log(np.array(expo_time_lst))
    NUM_IMAGE = len(expo_time_lst)
    NUM_PHOTO_USED = NUM_IMAGE
    SAMPLE_CNT = 2000

    # Load image    
    filename_lst = sorted(glob.glob(os.path.join(foldername, '*.jpg')))
    img_lst = []
    for filename in filename_lst[:NUM_PHOTO_USED]:
        img_lst.append(cv2.imread(filename))
      
    HEIGHT = img_lst[0].shape[0]
    WIDTH = img_lst[0].shape[1]

    # Calculate offset with MTB Alignment and align images
    offset_lst = MTB(img_lst, plot=False)

    # perform HDR
    irradiance = jpg_HDR(img_lst, offset_lst, expo_time_ln)

    # Save irradiance
    # .npy file
    irradiance = irradiance.astype('float32')
    np.save('Irradiance_JPG.npy', irradiance)

    # .hdr file
    img_max = np.max(irradiance)
    img_min = np.min(irradiance)
    irradiance /= img_max
    cv2.imwrite('Irradiance_JPG.hdr', irradiance)