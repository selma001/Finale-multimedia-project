from find import Find
import cv2

THRESHOLD = 50

img1 = cv2.imread("image072.png")
img2 = cv2.imread('image092.png')

choice = input("\nType de recherche ?\n -1) Lineaire\n -2) Logarithmique\n")

algorithm = {
    '1': Find.with_sliding,
    '2': Find.with_dichotomic,
}

algorithm[choice](img1, img2, THRESHOLD, DELTA=64)