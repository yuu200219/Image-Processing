import numpy as np
import cv2, os
from tqdm import tqdm
import matplotlib.pyplot as plt


OUTPUT_DIR = "./HW4_output_image/"
sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def sobel(img):
    rows, cols = img.shape
    res = np.zeros((rows, cols), np.uint8)
    for i in range(1, rows-1, 1):
        for j in range(1, cols-1, 1):
            f = np.array([[img[i-1][j-1], img[i-1][j], img[i-1][j+1]],
                         [img[i][j-1], img[i][j], img[i][j+1]],
                         [img[i+1][j-1], img[i+1][j], img[i+1][j+1]]])/255
            
            res_x = np.sum(np.multiply(sobel_x, f))*255
            res_y = np.sum(np.multiply(sobel_y, f))*255
            
            if res_x > 255:
                res_x = 255
            elif res_x < 0:
                res_x = 0
                
            if res_y > 255:
                res_y = 255
            elif res_y < 0:
                res_y = 0
                
            res[i:i+3,j:j+3] = res_x + res_y 
    
    return res

def plot(im, j, title):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, len(im), num=f"Sobel operation on {j}th image", constrained_layout=True)
    for i, val in enumerate(im):
        ax[i].set_title(title[i])
        ax[i].imshow(val)
        ax[i].axis("off")
    
    plt.savefig(OUTPUT_DIR + f"{j}.jpg")
    plt.show()

def main():
    im = []
    im.append(cv2.imread("./HW4_test_image/baboon.png"))
    im[0] = im[0][...,::-1]
    im.append(cv2.imread("./HW4_test_image/peppers.png"))
    im[1] = im[1][...,::-1]
    im.append(cv2.imread("./HW4_test_image/pool.png"))
    im[2] = im[2][...,::-1]

    result = []

    for t in range(3):
        rows, cols, d = im[t].shape
        res = np.zeros((rows, cols, 3), np.uint8)
        for k in tqdm(range(3), desc=f"Processing Sobel operation on {t}th image"):
            res[:,:,k] = sobel(im[t][:,:,k])
        result.append(res)
        
    for t in range(3):
        plot([im[t], result[t]], t, ["origin image", "afte Sobel operation"])
    
if __name__ == "__main__":
    main()