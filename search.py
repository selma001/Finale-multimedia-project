import numpy as np
from math import inf

class Search:

    def MSE(bloc_1, bloc_2):
        x, y = bloc_1.shape
        difference = np.square(bloc_1-bloc_2)
        return np.sum(difference)/(x*y)

    def dichotomique_search(target_bloc, searching_image, x, y, size=16, decalage=32) -> np.ndarray:

        pas = 16
        min_mse = inf

        coord_x = x+decalage
        coord_y = y+decalage

        min_x = coord_x
        min_y = coord_y

        min_bloc = None

        while pas >= 1:
            MOUVEMENTS = [(0, 0), (pas, 0), (-pas, 0), (0, pas), (-pas, 0),
                          (pas, pas), (-pas, -pas), (pas, -pas), (-pas, pas)]

            for move in MOUVEMENTS:
                a, s = move # x dans a et y dans s
                coord_x += a
                coord_y += s
                bloc = searching_image[coord_x:coord_x + size, coord_y:coord_y+size]

                if target_bloc.shape == bloc.shape:

                    mse = Search.MSE(target_bloc, bloc)
                    if  mse < min_mse:
                        min_mse = mse
                        min_x = coord_x
                        min_y = coord_y
                        min_bloc = bloc

                coord_x -= a
                coord_y -= s

            pas //= 2

        return min_bloc, min_x, min_y, min_mse

    
