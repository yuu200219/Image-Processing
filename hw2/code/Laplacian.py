import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt

mask_type = 4
OUTPUT_DIR = "./HW2_output_image/Laplacian/_" + str(mask_type) + "/"

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
    print("white: ", white)
    print("medium: ", medium)
    print("black: ", black)
    return mask_img, filtered_img
    
def plot(img, masked, filtered, title):
    fig, ax = plt.subplots(1, 3, figsize=(8, 5), num=title)   
    ax[0].imshow(img, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("original "+title)
    ax[1].imshow(masked, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("masked "+title)
    ax[2].imshow(filtered, cmap="gray")
    ax[2].axis("off")
    ax[2].set_title("After Laplacian operator")
    
    # ax[1, 0].hist(img.flatten(), 256, [0, 256])
    # ax[1, 1].hist(masked.flatten(), 256, [0, 256])
    # ax[1, 2].hist(filtered.flatten(), 256, [0, 256])
    plt.suptitle(title+" processed by Laplacian operator.")
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_DIR+title+".png")
    plt.show()

def main():
    img1 = cv2.imread("./HW2_test_image/blurry_moon.tif", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./HW2_test_image/skeleton_orig.bmp", cv2.IMREAD_GRAYSCALE)
    
    masked_1, filtered_1 = Laplacian(img1)
    masked_2, filtered_2 = Laplacian(img2)
    plot(img1, masked_1, filtered_1, "blurry_moon")
    plot(img2, masked_2, filtered_2, "skeleton")
    
if __name__ == '__main__':
    main()