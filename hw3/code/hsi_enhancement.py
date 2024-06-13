import cv2, os, math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

OUTPUT_DIR = "./HW3_output_image/hsi/"
mask_type = "-4"

def get_gray_level_num(img):
    rows, cols = img.shape
    n_k = OrderedDict((key, 0) for key in range(int(img.max())+1))
    # n_k = OrderedDict((key, 0) for key in range(256))
    
    # print(n_k)
    for i in range(rows):
        for j in range(cols):
            # print(i, j)
            # print(img[i][j])
            n_k[int(img[i][j])] += 1
    return n_k

def histogram_equal(img):
    # print(cls)
    rows, cols = img.shape
    n = cols * rows
    
    n_k = get_gray_level_num(img)
    s_img = np.zeros((rows, cols), np.uint8)
    p_r = OrderedDict((key, 0.0) for key in range(int(img.max())+1))
    s = OrderedDict((key, 0.0) for key in range(int(img.max())+1))
    L = int(img.max())+1
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
            s_img[i, j] = s[int(img[i,j])]*(L-1)
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
    
def plot(im, j, title):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, len(im), num="RGB space enhancement", constrained_layout=True)
    for i, val in enumerate(im):
        ax[i].set_title(title[i])
        ax[i].imshow(val)
        ax[i].axis("off")
    
    plt.savefig(OUTPUT_DIR + f"{j}.jpg")
    plt.show()
    
def toHSI(image):
    h_images = []
    s_images = []
    i_images = []
    for k in tqdm(range(4), desc="Converting RGB to HSI"):
        # print(k)
        rows, cols = image[k][0].shape
        H = np.zeros((rows, cols))
        S = np.zeros((rows, cols))
        I = ((image[k][0]/255 + image[k][1]/255 + image[k][2]/255)/3)
        for i in range(rows):
            for j in range(cols):
                R = image[k][0][i][j]/255
                G = image[k][1][i][j]/255
                B = image[k][2][i][j]/255
                
                S[i][j] = 1 - np.multiply((3/(R+G+B)), (min(R, G, B)))
                if R+G+B == 0:
                    S[i][j] = 0
                    
                
                
                den = np.sqrt((R-G)**2+(R-B)*(G-B))
                theta = np.arccos((0.5*((R-G)+(R-B)))/den)
                
                if B>G:
                    theta = 2*np.pi - theta
                if den == 0:
                    theta = 0
                    
                H[i][j] = theta/(2*np.pi)
                # print(theta)
        h_images.append(H*255)
        s_images.append(S*255)
        i_images.append(I*255)
        # hsi_image = np.zeros((rows, cols, 3), np.uint8)
        # hsi_image[:,:,0] = H*255
        # hsi_image[:,:,1] = S*255
        # hsi_image[:,:,2] = I*255
        # plot([H, S, I, hsi_image], k, ["Hue", "Saturation", "Intensity", "combine"])
    return h_images, s_images, i_images

def toRGB(h_images, s_images, i_images):
    final_images = []
    for k in tqdm(range(4), desc="Converting HSI to RGB"):
        rows, cols = h_images[k].shape
        rgb_image = np.zeros((rows, cols, 3))
        I = i_images[k]/255
        S = s_images[k]/255
        H = h_images[k]/255
        h = H*2*np.pi
        for i in range(rows):
            for j in range(cols):
                if h[i][j]>=0 and h[i][j]<2*np.pi/3: # >= 0 and <120
                    B = I[i][j]*(1-S[i][j])
                    R = I[i][j]*(1 + (S[i][j]*math.cos(h[i][j]))/(math.cos(np.pi/3-h[i][j])))
                    G = 3*I[i][j]-(R + B)
                elif h[i][j]>=2*np.pi/3 and h[i][j]<4*np.pi/3:
                    h_ = h[i][j] - 2*np.pi/3
                    R = I[i][j]*(1-S[i][j])
                    G = I[i][j]*(1 + (S[i][j]*math.cos(h_))/(math.cos(np.pi/3-h_)))
                    B = 3*I[i][j]-(R + G)
                else:
                    h_ = h[i][j] - 4*np.pi/3
                    G = I[i][j]*(1-S[i][j])
                    B = I[i][j]*(1 + (S[i][j]*math.cos(h_))/(math.cos(np.pi/3-h_)))
                    R = 3*I[i][j]-(G + B)
                rgb_image[i][j][0] = R
                rgb_image[i][j][1] = G
                rgb_image[i][j][2] = B
        final_images.append(rgb_image)
    return final_images

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
    image_res = []
    for j in range(4):
        for i in range(3):
            # print(j, i)
            # print(im[j][:,:,i])
            image[j].append(im[j][:,:,i])
            
    H, S, I = toHSI(image)

    image_res = []
    for j in tqdm(range(4), desc="processing histogram for each image in Intensity space"):
        res = histogram_equal(I[j])
        # print(res.shape)
        image_res.append(res)

    final_res = toRGB(H, S, image_res)

    for j in range(4):
        plot([im[j], final_res[j]], j, ["before enhancement", "after enhancement"])

if __name__ == "__main__":
    main()