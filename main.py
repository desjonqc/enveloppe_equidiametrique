import matplotlib.pyplot as plt
import numpy as np

import enveloppe
import geometrie
from affichage import ResultatAnimation, EvolutionAnimation
from enveloppe import Ensemble, Calculateur1
from opencl import OpenCLProgram

programme = OpenCLProgram("functions.cl")

def arc_simple():
    e = Ensemble(geometrie.arc(100, (100, 200), np.pi * 3 / 6, 300, 5))
    methode = Calculateur1(e)

    methode.afficher()

    resultats = methode.iteration_enveloppe_cl(programme, 0.1, 5)

    # for resultat in resultats:
    #     plt.imshow(resultat, cmap="Greys", interpolation="nearest")
    #     plt.show()

    anim = ResultatAnimation(e, resultats)
    anim.afficher()

def evolution_cercle():
    N = 60
    ensembles = [Ensemble(geometrie.arc(100, (150, 150), 2 * np.pi * i / N, 300, 5)) for i in range(N)]

    enveloppes = enveloppe.evolution(programme, Calculateur1, ensembles, 0.1)
    evolution = EvolutionAnimation(enveloppes, ensembles_superposes=ensembles)
    evolution.sauvegarder("evolution.gif")
    evolution.afficher()


if __name__ == "__main__":
    ensemble = Ensemble(geometrie.importer_image("formes/texas.gif"))

    methode = Calculateur1(ensemble)
    methode.afficher()
    figure = methode.enveloppe_equidiametrique_cl(programme, 0.3, reduite=False)

    plt.imshow((figure - ensemble.matrice_de_travail(0.3)), cmap="Greys", interpolation="nearest")
    plt.show()




