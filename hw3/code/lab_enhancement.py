import cv2, os, math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

OUTPUT_DIR = "./HW3_output_image/Lab/"
mask_type = "-4"

def histogram_equal(img):
    rows, cols = img.shape
    n = cols * rows
    n_k = np.zeros((256, ))
    s_img = np.zeros((rows, cols))
    s = np.zeros((256, ))
    
    for i in range(rows):
        for j in range(cols):
            n_k[int(img[i][j])] += 1
    
    for k in range(256):
        for j in range(k+1):
            s[k] += n_k[j]/n

    for i in range(rows):
        for j in range(cols): 
            s_img[i][j] = round(s[int(img[i][j])]*255)
    return s_img

def plot(im, j, title):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, len(im), num="RGB space enhancement", constrained_layout=True)
    for i, val in enumerate(im):
        ax[i].set_title(title[i])
        ax[i].imshow(val)
        ax[i].axis("off")
    
    plt.savefig(OUTPUT_DIR + f"{j}.jpg")
    plt.show()
    
# RGB2XYZ
# reference
# https://zh.wikipedia.org/wiki/CIE_1931%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4#%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C%E2%80%94_CIE_RGB%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4
def RGB2XYZ(im):
    trans = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])
    # trans = np.array([[0.5767309, 0.1855540, 0.1881852],
    #                   [0.2973769, 0.6273491, 0.0752741],
    #                   [0.0270343, 0.0706872, 0.9911085]])
    rows, cols = im[:,:,0].shape
    xyz = np.zeros((rows, cols, 3))
    for i in range(rows):
        for j in range(cols):
            R = im[i][j][0]/255
            G = im[i][j][1]/255
            B = im[i][j][2]/255
            rgb = np.array([R, G, B])
            xyz[i][j] = np.dot(trans,  rgb)*255
            
    return xyz
# XYZ2Lab
def f(t):
    if t > 0.008856:
        return t**(1/3)
    else:
        return 7.878* t + 16/116 
def XYZ2Lab(xyz):
    Xn = 0.950471
    Yn = 1.0000001
    Zn = 1.08883
    X = xyz[:,:,0]/255
    Y = xyz[:,:,1]/255
    Z = xyz[:,:,2]/255
    rows, cols = X.shape
    L = np.zeros((rows, cols))
    a = np.zeros((rows, cols))
    b = np.zeros((rows, cols))
    lab = np.zeros((rows, cols, 3))
    for i in range(rows):
        for j in range(cols):
            L[i][j] = round(116*f(Y[i][j]/Yn) - 16)
            a[i][j] = round(500*(f(X[i][j]/Xn)-f(Y[i][j]/Yn)))
            b[i][j] = round(200*(f(Y[i][j]/Yn) - f(Z[i][j]/Zn)))
            
    lab[:,:,0] = L
    lab[:,:,1] = a
    lab[:,:,2] = b
    return lab

def g(t):
    if t > 6/29:
        return t**3
    else:
        return 3*(6/29)**2*(t-4/29)
def Lab2XYZ(lab):
    Xn = 0.950471
    Yn = 1.0000001
    Zn = 1.08883
    L = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    rows, cols = L.shape
    X = np.zeros((rows, cols))
    Y = np.zeros((rows, cols))
    Z = np.zeros((rows, cols))
    fx = np.zeros((rows, cols))
    fy = np.zeros((rows, cols))
    fz = np.zeros((rows, cols))
    xyz = np.zeros((rows, cols, 3))
    for i in range(rows):
        for j in range(cols):
            fy[i][j] = (L[i][j]+16)/116 # fy
            fx[i][j] = fy[i][j] + a[i][j]/500 #fx
            fz[i][j] = fy[i][j] - b[i][j]/200 #fz
            
            Y[i][j] = Yn*g(fy[i][j])
            X[i][j] = Xn*g(fx[i][j])
            Z[i][j] = Zn*g(fz[i][j])
    xyz[:,:,0] = X
    xyz[:,:,1] = Y
    xyz[:,:,2] = Z
    
    return xyz

def XYZ2RGB(im):
    trans = np.array([[3.240479, -1.537150, -0.498535], 
                      [-0.969256, 1.875992, 0.041556],
                      [0.055648, -0.204043, 1.057311]])
    # trans = np.array([[2.043690, -0.5649464, -0.3446944],
    #                   [-0.9692660, 1.8760108, 0.0415560],
    #                   [0.0134474, -0.1183897, 1.0154096]])
    rows, cols = im[:,:,0].shape
    rgb = np.zeros((rows, cols, 3))
    
    for i in range(rows):
        for j in range(cols):
            X = im[i][j][0]
            Y = im[i][j][1]
            Z = im[i][j][2]
            xyz = np.array([X, Y, Z])
            val = np.dot(trans, xyz)
            rgb[i][j] = val
    return rgb
    
def main():
    im = []
    im.append(cv2.imread("./HW3_test_image/aloe.jpg"))
    im[0] = im[0][...,::-1]
    im.append(cv2.imread("./HW3_test_image/church.jpg"))
    im[1] = im[1][...,::-1]
    im.append(cv2.imread("./HW3_test_image/house.jpg"))
    im[2] = im[2][...,::-1]
    im.append(cv2.imread("./HW3_test_image/kitchen.jpg"))
    im[3] = im[3][...,::-1]
    
    xyz = []
    lab = []
    for k in tqdm(range(4), desc="Processing RGB2L*a*b"):
        rows, cols, d = im[k].shape
        #xyz.append(np.zeros((rows, cols, 3)))
        xyz.append(RGB2XYZ(im[k]))
        lab.append(XYZ2Lab(xyz[k]))
    L_enhance = []
    Lab_enhance = []
    
    for k in tqdm(range(4), desc="Processing histogram equalization"):
        L_enhance.append(histogram_equal(lab[k][:,:,0]))
        # L_enhance.append(cv2.equalizeHist(np.uint8(lab[k][:,:,0])))
        # a_enhance.append(histogram_equal(lab[k][:,:,1]))
        # b_enhance.append(histogram_equal(lab[k][:,:,2]))
        rows, cols = lab[k][:,:,0].shape
        Lab_enhance.append(np.zeros((rows, cols, 3)))

        Lab_enhance[k][:,:,0] = L_enhance[k]
        Lab_enhance[k][:,:,1] = lab[k][:,:,1]
        Lab_enhance[k][:,:,2] = lab[k][:,:,2]
        # Lab_enhance[k][:,:,1] = a_enhance[k]
        # Lab_enhance[k][:,:,2] = b_enhance[k]
    xyz_ = []
    rgb_ = []
    for k in tqdm(range(4), desc="Processing L*a*b to RGB"):
        # xyz_.append(Lab2XYZ(lab[k]))
        xyz_.append(Lab2XYZ(Lab_enhance[k]))
        rgb_.append(XYZ2RGB(xyz_[k]))
        
    for k in range(4):
        plot([im[k], rgb_[k]], k, ["before enhancement", "after enhancement"])

if __name__ == "__main__":
    main()