import cv2
import numpy as np
from tqdm import tqdm

TIME = 5000

class Fonctionalities:
    
    @staticmethod
    def extract_from(list, i, j):
        for x, y, k, l in list:
            if k == i and l == j:
                return x, y
        return i, j

    @staticmethod
    def show_predictions(image1, residue_image, list_of_blocs, bs, mode):
        filename = "./out/predicted_image"
        predicted_image = residue_image.copy()

        for x, y, i, j in tqdm(list_of_blocs):
            zone_one = predicted_image[i:i+bs, j:j+bs]
            zone_two = image1[x:x+bs, y:y+bs]
            if zone_one.shape == zone_two.shape:
                predicted_image[i:i+bs, j:j + bs] = image1[x:x+bs, y:y+bs]
            else:
                bs_i = zone_two.shape[0]
                bs_j = zone_two.shape[1]
                predicted_image[i:i+bs_i, j:j +
                                bs_j] = image1[x:x+bs, y:y+bs]

        cv2.imwrite(f"{filename}.{mode}.jpg", predicted_image)
        cv2.imshow("predicted_image", predicted_image)
        cv2.waitKey(TIME)

    @staticmethod
    def show_similarities(image, list_of_blocs, bs, mode):
        filename = "./out/similarities_image"
        for x, y, _, _ in tqdm(list_of_blocs):
            cv2.rectangle(image, (y, x, bs, bs), (0, 0, 0), 2)

        cv2.imwrite(f"{filename}.{mode}.jpg", image)
        cv2.imshow("similarities_between_images", image)
        cv2.waitKey(TIME)
        return image

    @staticmethod
    def show_residues(image2, list_of_blocs, bs, mode):
        filename = "./out/residues_image"
        residue_image = np.copy(image2)
        for _, _, x, y in tqdm(list_of_blocs):
            residue_image[x:x+bs, y:y+bs] = 0

        cv2.imwrite(f"{filename}.{mode}.jpg", residue_image)
        cv2.imshow("residues_between_images", residue_image)
        cv2.waitKey(TIME)
        return residue_image
    
    @staticmethod
    def process_both(im1, im2, DELTA, padding=False):
        grayImg1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        grayImg2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        if padding:
            grayImg1 = Fonctionalities.add_padding(grayImg1, DELTA)

        return grayImg1, grayImg2

    @staticmethod
    def add_padding(image, DELTA=64):
        image_with_padding = cv2.copyMakeBorder(
            image,
            DELTA,
            DELTA,
            DELTA,
            DELTA,
            cv2.BORDER_CONSTANT,
            None,
            0
        )
        return image_with_padding