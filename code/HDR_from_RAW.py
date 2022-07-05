import cv2  # For exporting .hdr file only
import numpy as np
import rawpy
import glob
import os
import copy
import sys


## =========== Demosaic =========== ##
def dem_bilinear(raw):
    (height, width) = raw.shape
    result = np.zeros((raw.shape[0], raw.shape[1], 3))
    
    # red
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            # Recover upper left blk
            result[i, j, 0] = raw[i, j]
    
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            # Recover upper right blk
            if i != height - 2:
                result[i + 1, j, 0] = result[i, j, 0] + result[i + 2, j, 0]
            else:
                result[i + 1, j, 0] = result[i, j, 0] + result[i, j, 0]
            
            # Recover lower left blk
            if j != width - 2:
                result[i, j + 1, 0] = result[i, j, 0] + result[i, j + 2, 0]
            else:
                result[i, j + 1, 0] = result[i, j, 0] + result[i, j, 0]

    for i in range(0, height, 2):
        for j in range(0, width, 2):
            # Recover lower right blk
            if i != height - 2:
                result[i + 1, j + 1, 0] = result[i, j + 1, 0] + result[i + 2, j + 1, 0]
            else:
                result[i + 1, j + 1, 0] = result[i, j + 1, 0] + result[i, j + 1, 0]
    
    # blue
    for i in range(1, height, 2):
        for j in range(1, width, 2):
            # Recover lower right blk
            result[i, j, 2] = raw[i, j]

    for i in range(1, height, 2):
        for j in range(1, width, 2):
            # Recover upper right blk
            if i != 1:
                result[i - 1, j, 2] = result[i, j, 2] + result[i - 2, j, 2]
            else:
                result[i - 1, j, 2] = result[i, j, 2] + result[i, j, 2]
            
            # Recover lower left blk
            if j != 1:
                result[i, j - 1, 2] = result[i, j, 2] + result[i, j - 2, 2]
            else:
                result[i, j - 1, 2] = result[i, j, 2] + result[i, j, 2]

    for i in range(1, height, 2):
        for j in range(1, width, 2):
            # Recover upper left blk
            if i != 1:
                result[i - 1, j - 1, 2] = result[i, j - 1, 2] + result[i - 2, j - 1, 2]
            else:
                result[i - 1, j - 1, 2] = result[i, j - 1, 2] + result[i, j - 1, 2]


    # green
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            result[i, j + 1, 1] = raw[i, j + 1]
            result[i + 1, j ,1] = raw[i + 1, j]
    
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            if i != 0:
                if j != 0:
                    result[i, j, 1] = result[i - 1, j, 1] + result[i + 1, j, 1] + result[i, j - 1, 1] + result[i, j + 1, 1]
                else:
                    result[i, j, 1] = result[i - 1, j, 1] + result[i + 1, j, 1] + result[i, j + 1, 1] + result[i, j + 1, 1]
            else:
                if j != 0:
                    result[i, j, 1] = result[i + 1, j, 1] + result[i + 1, j, 1] + result[i, j - 1, 1] + result[i, j + 1, 1]
                else:
                    result[i, j, 1] = result[i + 1, j, 1] + result[i + 1, j, 1] + result[i, j + 1, 1] + result[i, j + 1, 1]

            if i != height - 2:
                if j != width - 2:
                    result[i + 1, j + 1, 1] = result[i, j + 1, 1] + result[i + 2, j + 1, 1] + result[i + 1, j, 1] + result[i + 1, j + 2, 1]
                else:
                    result[i + 1, j + 1, 1] = result[i, j + 1, 1] + result[i + 2, j + 1, 1] + result[i + 1, j, 1] + result[i + 1, j, 1]
            else:
                if j != width - 2:
                    result[i + 1, j + 1, 1] = result[i, j + 1, 1] + result[i, j + 1, 1] + result[i + 1, j, 1] + result[i + 1, j + 2, 1]
                else:
                    result[i + 1, j + 1, 1] = result[i, j + 1, 1] + result[i, j + 1, 1] + result[i + 1, j, 1] + result[i + 1, j, 1]

    result[1::2, ::2, 0] = result[1::2, ::2, 0] // 2
    result[::2, 1::2, 0] = result[::2, 1::2, 0] // 2
    
    result[1::2, ::2, 2] = result[1::2, ::2, 2] // 2
    result[::2, 1::2, 2] = result[::2, 1::2, 2] // 2
    
    result[::2, ::2, 1:3] = result[::2, ::2, 1:3] // 4
    result[1::2, 1::2, 0:2] = result[1::2, 1::2, 0:2] // 4

    return result


def dem_half_res(raw):
    result = np.zeros((raw.shape[0] // 2, raw.shape[1] // 2, 3))
    result[:, :, 0] = raw[::2, ::2].copy()  # Red
    result[:, :, 1] = (raw[1::2, ::2].copy() + raw[::2, 1::2].copy()) // 2  # Green
    result[:, :, 2] = raw[1::2, 1::2].copy()  # Blue

    return result


def demosaic(raw, method):

    if method == 'bilinear' or method == 'freeman':
        result = dem_bilinear(raw)
        if method == 'freeman':
            from scipy import signal
            rg_diff = result[:, :, 0] - result[:, :, 1]
            bg_diff = result[:, :, 2] - result[:, :, 1]
            rg_medfilt = signal.medfilt2d(rg_diff, kernel_size=5)
            bg_medfilt = signal.medfilt2d(bg_diff, kernel_size=5)

            result[:, :, 0] = result[:, :, 1] + rg_medfilt
            result[:, :, 2] = result[:, :, 1] + bg_medfilt

    elif method == 'half-res':
        result = dem_half_res(raw)
    else:
        print('Please select an existing algorithm (half-res, bilinear, freeman)')
    return result.astype('uint16')


## ======== Inverse Mapping ======== ##
clip_14 = [0., 1488., 2688., 5184., 7580., 16692.]
span_14 = [744. / 1488., 300. / 1200., 312. / 2496., 150. / 2396., 285. / 9112.]
clip_11 = [256., 1000., 1300., 1612., 1762., 2047.]

def inverse(value):
    value = value - 512  # 512 is the black pixel value
    for i in range(5):
        if value < clip_14[i + 1]:
            break
    inverse_value = (value - clip_14[i]) * span_14[i] + clip_11[i]
    return inverse_value.astype('uint16')
    

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
    return offset_lst
    

## ========= HDR Estimation ========= ##
CLIP_SIZE = 256
RAW_MAX = 2047  # 11-bit value
RAW_MIN = 256
RAW_MEDIAN = round((RAW_MAX + RAW_MIN) / 2)
UPPER_BOUND = RAW_MAX - CLIP_SIZE
LOWER_BOUND = RAW_MIN + CLIP_SIZE
SPAN = (UPPER_BOUND - LOWER_BOUND) / 2

def debevec_weight(x):
    if x > UPPER_BOUND or x < LOWER_BOUND:
        return 0
    else:
        return np.abs(x - RAW_MEDIAN) / SPAN
        
        
def HDR(rgb_lst, expo_time_lst, offset_lst, NUM_PHOTO_USED):
    (h, w) = rgb_lst[0].shape[:2]
    irradiance = np.zeros(rgb_lst[0].shape, dtype=float)
    for i in range(h):
        for j in range(w):
            numerator = [0] * 3
            denominator = [0.0001] * 3

            for n in range(NUM_PHOTO_USED):
                coordinate = [i + offset_lst[n][0], j + offset_lst[n][1]]
                if coordinate[0] >= 0 and coordinate[0] < h and coordinate[1] >= 0 and coordinate[1] < w:
                    # Update value only if coordinate falls in legal range. 
                    for rgb in range(3):
                        pixel_value = rgb_lst[n][coordinate[0], coordinate[1], rgb]
                        weight = debevec_weight(pixel_value)
                        numerator[rgb] += expo_time_lst[n] * pixel_value * weight
                        denominator[rgb] += pow(expo_time_lst[n], 2) * weight

            irradiance[i][j][0] = numerator[0] / denominator[0]
            irradiance[i][j][1] = numerator[1] / denominator[1]
            irradiance[i][j][2] = numerator[2] / denominator[2]
            
    return irradiance
    
        
        
## ========= Main Function ========= ##
if __name__ == "__main__":

    foldername = 'data/raw/'
    expo_time_lst = [8, 4, 2, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    NUM_IMAGE = len(expo_time_lst)
    NUM_PHOTO_USED = NUM_IMAGE
    
    # Load image    
    filename_lst = sorted(glob.glob(os.path.join(foldername, '*.ARW')))[:NUM_PHOTO_USED]
    raw_lst = []
    for filename in filename_lst:
        raw = rawpy.imread(filename)
        raw_lst.append(raw.raw_image_visible.copy())
      
    HEIGHT = raw_lst[0].shape[0]
    WIDTH = raw_lst[0].shape[1]
    
    # Apply inverse mapping
    raw_inv_lst = []
    for idx in range(NUM_IMAGE):
        raw = raw_lst[idx].copy()
        raw_inv = np.zeros(raw.shape, dtype='uint16')
        for i in range(HEIGHT):
            for j in range(WIDTH):
                raw_inv[i, j] = inverse(raw[i, j])
      
        raw_inv_lst.append(raw_inv)
        
    # Apply demosaic
    rgb_lst = []
    for i in range(NUM_IMAGE):
        raw = raw_inv_lst[i]
        rgb = demosaic(raw, 'bilinear')
        rgb_lst.append(rgb)
        
    # Calculate offset with MTB Alignment
    offset_lst = MTB(rgb_lst, plot=False)
    
    # Perform HDR
    irradiance = HDR(rgb_lst, expo_time_lst, offset_lst, NUM_PHOTO_USED)
    
    # Apply white balance weights
    irradiance[:, :, 0] = irradiance[:, :, 0] * 2504. / 1024.
    irradiance[:, :, 2] = irradiance[:, :, 2] * 2640. / 1024.
    
    # Save irradiance
    # .npy file
    np.save('Irradiance_RAW.npy', irradiance)
    
    # .hdr file
    irradiance = irradiance.astype('float32')
    irradiance = irradiance[:, :, ::-1]
    
    cv2.imwrite('Irradiance_RAW.hdr', irradiance)