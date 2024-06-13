import cv2, os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

OUTPUT_DIR = "./HW3_output_image/rgb/"
mask_type = "-4"

def get_gray_level_num(img):
    rows, cols = img.shape
    n_k = OrderedDict((key, 0) for key in range(256))
    # n_k = OrderedDict((key, 0) for key in range(256))
    
    # print(n_k)
    for i in range(rows):
        for j in range(cols):
            # print(i, j)
            # print(img[i][j])
            n_k[img[i][j]] += 1
    return n_k

def histogram_equal(img):
    # print(cls)
    rows, cols = img.shape
    n = cols * rows
    
    n_k = get_gray_level_num(img)
    s_img = np.zeros((rows, cols), np.uint8)
    p_r = OrderedDict((key, 0.0) for key in range(256))
    s = OrderedDict((key, 0.0) for key in range(256))
    L = 256
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
    return s_img

def Laplacian(img):
    black = 0
    white = 0
    medium = 0
    rows, cols = img.shape
    mask_img = np.zeros((rows, cols), np.uint8)
    filtered_img = np.zeros((rows, cols), np.uint8)
    if mask_type == "-4":
        mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif mask_type == "4":
        mask = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
    elif mask_type == "-8":
        mask = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
    elif mask_type == "8":
        mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    else:
        print("the filter type dosen't exist")
        exit()
    for i in range(1, rows-1, 1):
        for j in range(1, cols-1, 1):
            f = np.array([[img[i-1][j-1], img[i-1][j], img[i-1][j+1]],
                         [img[i][j-1], img[i][j], img[i][j+1]],
                         [img[i+1][j-1], img[i+1][j], img[i+1][j+1]]])
            res = np.sum(np.multiply(f, mask))
            
            
            if res > 255:
                mask_img[i][j] = 255
                white += 1
            elif res < 0:
                mask_img[i][j] = 0
                black += 1
            else:
                mask_img[i][j] = res
                medium += 1
                
            if res < 0:
                if img[i][j] - res > 255:
                    filtered_img[i][j] = 255
                elif img[i][j] - res < 0:
                    filtered_img[i][j] = 0
                else:
                    filtered_img[i][j] = img[i][j] - res
                    # filtered_img = res
            else:
                if img[i][j] + res < 0:
                    filtered_img[i][j] = 0
                elif img[i][j] + res > 255:
                    filtered_img[i][j] = 255
                else:
                    filtered_img[i][j] = img[i][j] + res
                    # filtered_img = res
    # print("white: ", white)
    # print("medium: ", medium)
    # print("black: ", black)
    return filtered_img
    
def plot(im1, im2, j):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, num="RGB space enhancement", constrained_layout=True)
    ax[0].set_title("before enhancement")
    ax[1].set_title("after enhancement")
    ax[0].imshow(im1)
    ax[1].imshow(im2)
    ax[0].axis("off")
    ax[1].axis("off")
    plt.savefig(OUTPUT_DIR + f"{j}.jpg")
    plt.show()
        
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
    
    image = [[], [], [], []]
    image_res = [[], [], [], []]
    for j in range(4):
        for i in range(3):
            # print(j, i)
            # print(im[j][:,:,i])
            image[j].append(im[j][:,:,i])
    
    for j in tqdm(range(4), desc="processing histogram for each image in RGB"):
        for i, val in enumerate(image[j]):
            res = histogram_equal(val)
            image_res[j].append(res)
    
    final_res = [[], [], [], []]
    for j in range(4):
        rows, cols = image[j][0].shape
        final_res[j] = np.zeros((rows, cols, 3), np.uint8)
    for j in range(4):
        for i in range(3):
            final_res[j][:,:,i] = image_res[j][i]
    
    plt.imshow(final_res[2])
    plt.show()
    # for j in range(4):
    #    plot(im[j], final_res[j], j)
    
    #print((image_res[:,:,:]).shape)
    
if __name__ == "__main__":
    main()
