import cv2
import numpy as np

TIME = 5000

class Fonctionalities:


    @staticmethod
    def show_residues(image2, Lblocs, size, mode):
        filename = "./out/residues_image"
        residue_image = np.copy(image2)
        for _, _, x, y in Lblocs:
            residue_image[x:x+size, y:y+size] = 0

        cv2.imwrite(f"{filename}.{mode}.jpg", residue_image)
        cv2.imshow("residues_between_images", residue_image)
        cv2.waitKey(TIME)
        return residue_image

