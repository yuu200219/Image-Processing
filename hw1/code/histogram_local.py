import cv2, os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
# local approach

OUTPUT_DIR = "./HW1_output_image/local/"

def histogram_equal(n_k, n, rows, cols, img):
    s_img = np.zeros((rows, cols), np.uint8)
    p_r = OrderedDict((key, 0.0) for key in range(img.max()+1))
    s = OrderedDict((key, 0.0) for key in range(img.max()+1))
    L = img.max()+1
    # p_r: normalize gray level
    # key: gray_level_th, value: normalize histogram
    for k in range(L):
        p_r[k] = n_k[k] / n
    # transformation
    for k in range(L):
        for j in range(k):
            s[k] += p_r[j]

    for i in range(rows):
        for j in range(cols): 
            s_img[i, j] = s[img[i,j]]*(L-1)
    # print(s_img.shape)
    return s_img
    
def get_gray_level_num(img, rows, cols):
    n_k = OrderedDict((key, 0) for key in range(256))
    # print(n_k)
    for i in range(rows):
        for j in range(cols):
            n_k[img[i,j]] += 1
    return n_k

def slicing(img, div, interval, cls):
    fig_1, ax1 = plt.subplots(div, div, num="slice image")
    fig_1.suptitle("slice image")
    
    slice_img = np.zeros((div, div, interval, interval), np.uint8)

    i = 0
    j = 0
    for x in range(0, 256, interval):
        for y in range(0, 256, interval):
            slice_img[i, j] = img[x:x+interval, y:y+interval]
            ax1[i,j].imshow(slice_img[i, j], cmap="gray")
            ax1[i,j].set_xticks([])
            ax1[i,j].set_yticks([])
            j += 1
        j = 0
        i += 1
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_DIR + "slice_"+cls)
    plt.show()
    return slice_img

def combine_slice_image(slice_img, div, interval):
    max_width = interval * div
    max_height = interval * div

    # Create a new blank image to paste the images onto
    concatenated_image = np.zeros((max_height, max_width), dtype=np.uint8)
    norm_slice_img = [(slice_img[i, j] - np.min(slice_img[i,j]))*255 / (np.max(slice_img[i,j]) - np.min(slice_img[i,j])) for i in range(div) for j in range(div)]
    
    for i in range(div):
        for j in range(div):
            row_start = i * interval
            row_end = (i + 1) * interval
            col_start = j * interval
            col_end = (j + 1) * interval

            image = norm_slice_img[i*div + j]  # Access the corresponding sliced image
            concatenated_image[row_start:row_end, col_start:col_end] = image
    return concatenated_image

def sub_plot(img, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    n_k = get_gray_level_num(img, img.shape[0], img.shape[1])
    x = 0 + np.arange(256)
    ax[0].imshow(img, cmap="gray")
    #ax[1].hist(img.flatten(), 256, [0, 256])
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
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        # ax[i, j+1].hist(img[i].flatten(), 256, [0,256])
        ax[i, j+1].bar(x, list(n_k.values())) 
    j += 2
    for i in range(rows):
        n_k = get_gray_level_num(img[i], img[i].shape[0], img[i].shape[1])
        ax[i, j].imshow(s_img[i], cmap="gray")
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        # ax[i, j+1].hist(s_img[i].flatten(), 256, [0,256])
        ax[i, j+1].bar(x, list(n_k.values()))
        
    plt.subplots_adjust(left=0.1, right=0.9, top=2.5, bottom=0.1)
    plt.suptitle(title)
    plt.tight_layout()  # Adjust subplot layout to prevent overlap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_DIR + title + ".png")
    plt.show()

def main():
    in_img_1 = './HW1_test_image/Lena.bmp'
    in_img_2 = './HW1_test_image/Peppers.bmp'
    img_1 = cv2.imread(in_img_1, 0)
    img_2 = cv2.imread(in_img_2, 0)
    rows_1, cols_1 = img_1.shape
    rows_2, cols_2 = img_2.shape
    n_1 = rows_1 * cols_1
    n_2 = rows_2 * cols_2
    div = 4
    interval = int(rows_1/div)
    # get the number of pixel for each gray level, n_k
    # h(r_k) = n_k
    # local
    slice_img = []
    slice_img.append(slicing(img_1, div, interval, "Lena"))
    slice_img.append(slicing(img_2, div, interval, "Peppers"))

    # print(img_1.shape)
    
    # get grey level num for each block
    n_k_k = np.zeros((2, div, div), OrderedDict)
    s_slice_img = np.zeros((2, div, div, interval, interval))
            
    # histogram equalization
    cls = ["Lena", "Peppers"]
    for k in range(2):
        
        for i in range(div):
            for j in range(div):
                r, c = slice_img[k][i, j].shape
                n_k_k[k][i, j] = get_gray_level_num(slice_img[k][i, j], r, c)      
        fig_2, ax2 = plt.subplots(div, div, num="slice image after hist_eq")
        fig_2.suptitle("slice image afte hist_eq")
        for i in range(div):
            for j in range(div): 
                r, c = slice_img[k][i, j].shape
                s_slice_img[k][i, j] = histogram_equal(n_k_k[k][i, j], int(r*c), r, c, slice_img[k][i, j])
                ax2[i,j].imshow(s_slice_img[k][i, j], cmap='gray')
                ax2[i,j].set_xticks([])
                ax2[i,j].set_yticks([])
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(OUTPUT_DIR + "hist_slice_" + cls[k] + ".png")
        plt.savefig("hist_slice_" + cls[k])
        plt.show()

    res_img_1 = combine_slice_image(s_slice_img[0], div, interval)
    res_img_2 = combine_slice_image(s_slice_img[1], div, interval)
    sub_plot(res_img_1, "Lena after hist_eq")
    sub_plot(res_img_2, "Peppers after hist_eq")
    plot([img_1, img_2], [res_img_1, res_img_2], "local histogram equalization")
    # sub_plot(s_img_1, "Lena after local histogram equalization")
    # sub_plot(s_img_2, "Peppers after local histogram equalization")
    # plot([img_1, img_2], [s_img_1, s_img_2], "local histogram equalization")
    # res_img = np.zeros((1, 256))
    # for i in range(16):
    #     row_img = s_slice_img[i, 0]
    #     for j in range(1, 16, 1):
    #         row_img = np.concatenate((s_slice_img[i, j], row_img), axis=1)  
    #     if i == 0:
    #         res_img = row_img
    #     res_img = np.concatenate((res_img, row_img), axis=0)
    #     ax3[i].imshow(res_img, cmap="gray")
    
    # plt.show()
    # plt.tight_layout()
    # plt.show()
    # cv2.imshow('original_img', img)
    # cv2.imshow('final_img', s_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()