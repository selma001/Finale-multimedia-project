import cv2
import numpy as np
from tqdm import tqdm

TIME = 5000

class Fonctionalities:

    def show_similarities(image, Lblocs, size, mode):
        filename = "./out/similarities_image"
        for x, y, _, _ in tqdm(Lblocs):
            cv2.rectangle(image, (y, x, size, size), (0, 0, 0), 2)

        cv2.imwrite(f"{filename}.{mode}.jpg", image)
        cv2.imshow("similarities_between_images", image)
        cv2.waitKey(TIME)
        return image

    def show_residues(image2, Lblocs, size, mode):
        filename = "./out/residues_image"
        residue_image = np.copy(image2)
        for _, _, x, y in tqdm(Lblocs):
            residue_image[x:x+size, y:y+size] = 0

        cv2.imwrite(f"{filename}.{mode}.jpg", residue_image)
        cv2.imshow("residues_between_images", residue_image)
        cv2.waitKey(TIME)
        return residue_image
