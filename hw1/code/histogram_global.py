import cv2, os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
# global approach

OUTPUT_DIR = "./HW1_output_image/global/"

def histogram_equal(n_k, n, rows, cols, img, cls):
    # print(cls)
    s_img = np.zeros((rows, cols), np.uint8)
    p_r = OrderedDict((key, 0.0) for key in range(img.max()+1))
    s = OrderedDict((key, 0.0) for key in range(img.max()+1))
    L = img.max()+1
    # print(L)
    # p_r: normalize gray level
    # key: gray_level_th, value: normalize histogram
    for k in range(L):
        p_r[k] = n_k[k] / n
    # transformation
    for k in range(L):
        for j in range(k):
            s[k] += p_r[j]
    # print(s)
    for i in range(rows):
        for j in range(cols): 
            s_img[i, j] = s[img[i,j]]*(L-1)
    # print(s_img)
    # print(s_img.shape)
    return s_img
    
def get_gray_level_num(img, rows, cols):
    # n_k = OrderedDict((key, 0) for key in range(img.max()+1))
    n_k = OrderedDict((key, 0) for key in range(256))
    # print(n_k)
    for i in range(rows):
        for j in range(cols):
            n_k[img[i,j]] += 1
    return n_k

def sub_plot(img, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    n_k = get_gray_level_num(img, img.shape[0], img.shape[1])
    x = 0 + np.arange(256)
    ax[0].imshow(img, cmap='gray')
    # ax[1].hist(img.flatten(), 256, [0, 256])
    ax[1].bar(x, list(n_k.values())) 
    plt.suptitle(title)
    plt.tight_layout()  # Adjust subplot layout to prevent overlap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_DIR + title+".png")
    plt.show()
    
def plot(img, s_img, title):
    rows = 2
    cols = 4
    
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6),num=title)
    x = 0 + np.arange(256)
    j = 0
    for i in range(rows):
        n_k = get_gray_level_num(img[i], img[i].shape[0], img[i].shape[1])
        ax[i, j].imshow(img[i], cmap="gray")
        # ax[i, j+1].hist(img[i].flatten(), 256, [0,256]) 
        ax[i, j+1].bar(x, list(n_k.values()))       
    j += 2
    for i in range(rows):
        ax[i, j].imshow(s_img[i], cmap="gray")
        # ax[i, j+1].hist(s_img[i].flatten(), 256, [0,256])
        ax[i, j+1].bar(x, list(n_k.values())) 
         
    plt.subplots_adjust(left=0.1, right=0.9, top=2.5, bottom=0.1)
    plt.suptitle(title)
    plt.tight_layout()  # Adjust subplot layout to prevent overlap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_DIR + title + ".png")
    plt.show()
    
    
def main():
    # read image
    in_img_1 = './HW1_test_image/Lena.bmp'
    in_img_2 = './HW1_test_image/Peppers.bmp'
    img_1 = cv2.imread(in_img_1, 0)
    img_2 = cv2.imread(in_img_2, 0)
    rows_1, cols_1 = img_1.shape
    rows_2, cols_2 = img_2.shape
    n_1 = rows_1 * cols_1
    n_2 = rows_2 * cols_2
    
    # get the number of pixel for each gray level, n_k
    # h(r_k) = n_k
    n_k_1 = get_gray_level_num(img_1, rows_1, cols_1)
    n_k_2 = get_gray_level_num(img_2, rows_2, cols_2)
    
    # histogram
    
    # histogram equalization
    s_img_1 = histogram_equal(n_k_1, n_1, rows_1, cols_1, img_1, "Lena")
    s_img_2 = histogram_equal(n_k_2, n_2, rows_2, cols_2, img_2, "Peppers")
    
    #plot image and histogram
    sub_plot(img_1, "Lena before histogram equalization")
    sub_plot(img_2, "Peppers beore histogram equalization")
    sub_plot(s_img_1, "Lena after histogram equalization")
    sub_plot(s_img_2, "Peppers after histogram equalization")
    plot([img_1, img_2], [s_img_1, s_img_2], "global histogram equalization")
    
if __name__ == '__main__':
    main()