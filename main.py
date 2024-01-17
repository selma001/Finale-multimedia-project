import numpy as np
from math import inf
from fonctionalities import Fonctionalities
import time
from tqdm import tqdm
from search import Search
import cv2

class Find:

    def with_dichotomic(im1, im2, threshold=50, size=16, decalage=64):

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
        list_of_blocs = []
        x1, y1 = image2.shape

        print("[Searching]...")
        start = time.time()
        for i in tqdm(range(0, x1, size)):
            for j in range(0, y1, size):
                target_bloc = image2[i:i+size, j:j+size]
                if target_bloc.shape == (size, size):
                    _, x, y, min_mse = Search.dichotomique_search(
                        target_bloc, image1, i, j, size, decalage)
                    if min_mse < threshold:
                        list_of_blocs.append((x-decalage, y-decalage, i, j))

        end = time.time()

        print(f"[Result] in {end-start}s")

        print(f"[Preparing] image similarities ...")
        Fonctionalities.show_similarities(im1.copy(), list_of_blocs, size, mode="log")
        print(f"[Done]")

        print(f"[Preparing] image residues ...")
        residue_image = Fonctionalities.show_residues(im2, list_of_blocs, size, mode="log")
        print(f"[Done]")

        
        print(f"[Finish]") 


#---- main test ----#
THRESHOLD = 50
img1 = cv2.imread("image072.png")
img2 = cv2.imread('image092.png')

Find.with_dichotomic(img1, img2, THRESHOLD)