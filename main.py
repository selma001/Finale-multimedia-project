import numpy as np
from fonctionalities import Fonctionalities
import time
from search import Search
import cv2

class Find:

    def with_dichotomic(im1, im2, threshold=50, size=16, decalage=32):

        image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        image1 = cv2.copyMakeBorder(
            image1,
            decalage,
            decalage,
            decalage,
            decalage,
            cv2.BORDER_CONSTANT,
            None,
            0
        )
        Lblocs = []
        x1, y1 = image2.shape

        print("Searching in process ...")
        start = time.time()

        
        for i in range(0, x1, size):
            for j in range(0, y1, size):
                target_bloc = image2[i:i+size, j:j+size]
                if target_bloc.shape == (size, size):
                    _, x, y, min_mse = Search.dichotomique_search(target_bloc, image1, i, j, size, decalage)
                    if min_mse < threshold:
                        Lblocs.append((x-decalage, y-decalage, i, j))

        end = time.time()

        print(f"Result in {end-start}s")


        print(f" image residues in process ...")
        residue_image = Fonctionalities.show_residues(im2, Lblocs, size, mode="log")
        print(f" residues DONE!")

        print(f"END of the search") 

#---- main test ----#
THRESHOLD = 50
img1 = cv2.imread("image072.png")
img2 = cv2.imread('image092.png')

Find.with_dichotomic(img1, img2, THRESHOLD)
