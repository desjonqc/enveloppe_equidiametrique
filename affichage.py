import numpy as np
from matplotlib import pyplot as plt, animation

import enveloppe
from enveloppe import Ensemble


class ResultatAnimation:
    """ Crée une animation à partir d'un ensemble et d'une liste de matrices """

    def __init__(self, ensemble: Ensemble, figures: list[np.ndarray], interval=500):
        self.figures = figures
        fig, ax = plt.subplots()
        self.container = [[ax.imshow(ensemble.elements, cmap="Greys", interpolation="nearest", animated=True)]]

        for i in range(len(figures)):
            figure = figures[i]
            self.container.append([ax.imshow(figure, cmap="Greys", interpolation="nearest", animated=True)])

        self.animation = animation.ArtistAnimation(fig, self.container, interval, blit=True)

    @staticmethod
    def afficher():
        plt.show()

    def sauvegarder(self, nom_fichier: str):
        self.animation.save(nom_fichier, writer="imagemagick")


class EvolutionAnimation:
    """ Crée une animation à partir d'une liste d'enveloppes """

    def __init__(self, figures: list[np.ndarray], ensembles_superposes: list[Ensemble] = None, interval=100):
        self.figures = figures
        fig, ax = plt.subplots()
        self.container = []

        for i in range(len(figures)):
            figure = figures[i]
            if ensembles_superposes is not None:
                matrice_reduite = enveloppe.reduire_matrice(ensembles_superposes[i].elements)  # Mise à l'échelle
                matrice_normalisee = Ensemble(matrice_reduite).matrice_de_travail(0)
                self.container.append(
                    [ax.imshow(figure + matrice_normalisee, cmap="Greys", interpolation="nearest", animated=True)])
            else:
                self.container.append([ax.imshow(figure, cmap="Greys", interpolation="nearest", animated=True)])

        self.animation = animation.ArtistAnimation(fig, self.container, interval, blit=True)

    @staticmethod
    def afficher():
        plt.show()

    def sauvegarder(self, nom_fichier: str):
        self.animation.save(nom_fichier, writer="imagemagick")
