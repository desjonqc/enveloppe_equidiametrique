import time

import numpy as np
import matplotlib.pyplot as plt

from opencl import OpenCLProgram, CLField, cl

def norme(a: tuple, b: tuple) -> float:
    """ Calcule la norme entre deux points a et b. """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def reduire_matrice(matrice: np.ndarray, marge=0.0) -> np.ndarray:
    """
    Transforme une matrice de travail (élargie pour le calcul de l'enveloppe) en une matrice compressée (d'un tiers)
     et supprime la marge appliquée à la matrice de départ.
    :param matrice: La matrice à réduire
    :param marge: La marge appliquée à la matrice avant le triplement de taille
    :return: la matrice réduite.
    """
    taille = len(matrice)
    nouvelle_matrice = np.zeros((taille // 3, taille // 3))
    surplus = int((taille / 3) * (marge / (2 * marge + 1)))  # Delta est calculée sur la matrice centrale
    print("Démarrage de la réduction de la matrice")

    precision = 1
    for i in range(0, taille, 3):
        for j in range(0, taille, 3):
            moyenne = np.mean(matrice[max(0, i - precision):min(int(i + 3 + precision), taille - 1),
                              max(0, int(j - precision)):min(int(j + 3 + precision), taille - 1)])
            nouvelle_matrice[i // 3, j // 3] = 1 if moyenne != 1 and moyenne > 2 / (precision + 3) ** 2 else 0
    print("Terminé.")
    return nouvelle_matrice[surplus:taille // 3 - surplus, surplus:taille // 3 - surplus]


def reduire_matrice_cl(programme: OpenCLProgram, taille: int, matrice_buffer, marge=0.0) -> np.ndarray:
    """
    Même fonction que reduire_matrice mais en utilisant OpenCL.
    :param programme: L'objet de gestion d'OpenCL
    :param taille: la taille de la matrice
    :param matrice_buffer: le buffer OpenCL contenant la matrice
    :param marge: la marge initialement appliquée à la matrice avant le triplement de taille
    :return: la matrice réduite.
    """
    surplus = int((taille / 3) * (marge / (2 * marge + 1)))  # Delta est calculée sur la matrice centrale
    nouvelle_matrice = np.zeros((taille // 3 - 2 * surplus, taille // 3 - 2 * surplus))

    pas = taille / (taille / 3 - 2 * surplus)
    precision = 3

    nouvelle_matrice_field = CLField(nouvelle_matrice)
    nouvelle_matrice_buffer = nouvelle_matrice_field.convert_to_cl(programme.ctx,
                                                                   CLField.mf.WRITE_ONLY | CLField.mf.USE_HOST_PTR)

    print("Démarrage de la réduction de la matrice")
    programme.call_function("reduce_matrix", (int(taille / pas), int(taille / pas)), matrice_buffer,
                            nouvelle_matrice_buffer, np.int32(taille), np.int32(len(nouvelle_matrice)),
                            np.int32(precision), np.float32(pas))
    print("Fin de la réduction de la matrice")

    return nouvelle_matrice_field.retrieve_from_cl(programme.queue, nouvelle_matrice_buffer)


class Ensemble:
    """ Représente un ensemble de points fini donc borné, et connexe (supposé). """
    def __init__(self, elements: np.ndarray):
        """
        :param elements: Tableau numpy représentant les points dans l'espace : 1 pour un point, 0 pour un vide.
        """
        self.elements = elements
        self.N = len(self.elements[self.elements == 1])  # Nombre de points de l'ensemble

    def representation_matricielle(self, marge: float) -> np.ndarray:
        """
        Calcule la matrice avec la marge appliquée.
        :param marge: La marge de pixel à appliquer en pourcentage.
        :return: La matrice avec la marge appliquée.
        """
        taille = len(self.elements)
        offset = int(taille * marge)
        if offset > 0:
            return np.block(
                [[np.zeros((offset, offset)), np.zeros((offset, taille)), np.zeros((offset, offset))],
                 [np.zeros((taille, offset)), self.elements, np.zeros((taille, offset))],
                 [np.zeros((offset, offset)), np.zeros((offset, taille)), np.zeros((offset, offset))]])
        else:
            return self.elements

    def matrice_de_travail(self, marge) -> np.ndarray:
        """
        Calcule la matrice de travail pour le calcul de l'enveloppe équidiamétrique. On applique à l'ensemble la marge
        donnée, puis on triple la taille de la matrice pour pouvoir appliquer la méthode de calcul de l'enveloppe.
        :param marge: La marge à appliquer à la matrice.
        :return: La matrice de travail.
        """
        rep = self.representation_matricielle(marge)
        taille = len(rep)
        return np.block([[np.zeros((taille, taille)), np.zeros((taille, taille)), np.zeros((taille, taille))],
                         [np.zeros((taille, taille)), rep, np.zeros((taille, taille))],
                         [np.zeros((taille, taille)), np.zeros((taille, taille)), np.zeros((taille, taille))]])

    def calc_delta(self) -> np.ndarray:
        """
        Calcule le tableau des distances maximales entre les points de l'ensemble. Version utilisant uniquement Python.
        :return: Le tableau des distances maximales.
        """
        delta = np.array([(i, j, 0) for i in range(len(self.elements)) for j in range(len(self.elements)) if
                          self.elements[i, j] == 1])
        for k in range(len(delta)):
            i, j, _ = delta[k]
            delta[k] = i, j, np.max([norme((i, j), (i2, j2)) for i2, j2, _ in delta])

        return delta

    def calc_delta_cl(self, programme: OpenCLProgram) -> np.ndarray:
        """
        Calcule le tableau des distances maximales entre les points de l'ensemble. Version utilisant OpenCL peu efficace :
        On parcourt tous les points de l'ensemble une fois et on calcule en parallèle la distance entre ce point et
        tous les autres points de l'ensemble.
        :param programme: L'objet de gestion d'OpenCL.
        :return: Le tableau des distances maximales.
        """
        delta = np.array([(i, j, 0) for i in range(len(self.elements)) for j in range(len(self.elements)) if
                          self.elements[i, j] == 1])

        delta_field = CLField(np.array(delta, dtype=np.float32))
        delta_buffer = delta_field.convert_to_cl(programme.ctx, CLField.mf.READ_WRITE | CLField.mf.USE_HOST_PTR)
        for i in range(len(delta)):
            programme.call_function("calc_delta", (len(delta),), delta_buffer, np.int32(i))

        return_value = delta_field.retrieve_from_cl(programme.queue, delta_buffer)
        return return_value

    def calc_delta_cl_efficace(self, programme: OpenCLProgram) -> np.ndarray:
        """
        Calcule le tableau des distances entre les points de l'ensemble. Version utilisant OpenCL efficace :
        On calcule en parallèle la distance entre chaque point de l'ensemble et tous les autres points de l'ensemble.
        L'objet rendu par la fonction OpenCL est une matrice telle que M[i, j] est la distance entre le point i et le
        point j (indice dans le tableau positions).

        On calcule alors le max de chaque ligne de cette matrice pour obtenir le tableau des distances maximales.

        :param programme: L'objet de gestion d'OpenCL.
        :return: Le tableau des distances maximales.
        """
        positions = np.array([(i, j) for i in range(len(self.elements)) for j in range(len(self.elements)) if
                              self.elements[i, j] == 1])

        delta_matrix = np.zeros((len(positions), len(positions)), dtype=np.float32)

        pos_field = CLField(np.array(positions, dtype=np.float32))
        pos_buffer = pos_field.convert_to_cl(programme.ctx, CLField.mf.READ_ONLY | CLField.mf.USE_HOST_PTR)
        delta_matrix_buffer = cl.Buffer(programme.ctx, CLField.mf.WRITE_ONLY, delta_matrix.nbytes)

        programme.call_function("calc_delta_eff", (len(positions), len(positions)), pos_buffer, delta_matrix_buffer,
                                np.int32(len(positions)))

        cl.enqueue_copy(programme.queue, delta_matrix, delta_matrix_buffer)

        delta_vect = np.max(delta_matrix, axis=1)
        delta_vect_buffer = CLField(delta_vect).convert_to_cl(programme.ctx,
                                                              CLField.mf.READ_ONLY | CLField.mf.USE_HOST_PTR)
        delta = np.zeros((len(positions), 3), dtype=np.float32)
        delta_buffer = cl.Buffer(programme.ctx, CLField.mf.WRITE_ONLY, delta.nbytes)

        programme.call_function("format_delta", (len(positions),), delta_buffer, pos_buffer, delta_vect_buffer,
                                np.int32(len(positions)))

        delta = np.zeros((len(positions), 3), dtype=np.float32)
        cl.enqueue_copy(programme.queue, delta, delta_buffer)

        return delta

    def afficher(self):
        """ Affiche l'ensemble de points dans un graphique. """
        plt.imshow(self.elements, cmap="Greys", interpolation="nearest")
        plt.show()

    def __len__(self):
        """
        :return: Le nombre de points de l'ensemble.
        """
        return self.N


class CalculateurEnveloppe:
    """ Classe abstraite permettant de calculer l'enveloppe équidiamétrique d'un ensemble de points. """
    def __init__(self, ensemble: Ensemble):
        """
        :param ensemble: L'ensemble de points dont on veut calculer l'enveloppe.
        """
        self.ensemble = ensemble

    def enveloppe_equidiametrique(self, marge: float, reduite=True) -> np.ndarray:
        """
        Calcule l'enveloppe équidiamétrique de l'ensemble de points.
        :param marge: La marge à appliquer à l'ensemble de points pour calculer la matrice de travail.
        :param reduite: Si True, la matrice de travail est réduite à la taille initiale. Sinon, elle est renvoyée telle
        quelle, environ 3 fois plus grande.
        :return: La matrice de l'enveloppe équidiamétrique.
        """
        pass

    def enveloppe_equidiametrique_cl(self, programme: OpenCLProgram, marge: float, reduite=True) -> np.ndarray:
        """
        Calcule l'enveloppe équidiamétrique de l'ensemble de points en utilisant OpenCL.
        :param programme: L'objet de gestion d'OpenCL.
        :param marge: La marge à appliquer à l'ensemble de points pour calculer la matrice de travail.
        :param reduite: Si True, la matrice de travail est réduite à la taille initiale. Sinon, elle est renvoyée telle
        quelle, environ 3 fois plus grande.
        :return: La matrice de l'enveloppe équidiamétrique.
        """
        pass

    def _iteration_enveloppe(self, marge: float, nombre_iterations: int, func) -> list[np.ndarray]:
        """
        Itère le calcul de l'enveloppe équidiamétrique. Fonction interne : ne pas utiliser directement.
        :param marge: La marge à appliquer à l'ensemble de points pour calculer la matrice de travail.
        :param nombre_iterations: Le nombre d'itérations à effectuer.
        :param func: La fonction de calcul de l'enveloppe équidiamétrique à appliquer (entre Python et OpenCL).
        :return: La liste des enveloppes équidiamétriques calculées.
        """
        ensemble = self.ensemble
        enveloppes = []
        for i in range(nombre_iterations):
            print(f"Iteration {i + 1} / {nombre_iterations}...")
            t = time.time()
            enveloppe = func(marge)
            self.ensemble = Ensemble(enveloppe)
            enveloppes.append(enveloppe)
            print(f"Fin de l'itération : {time.time() - t} s")
        self.ensemble = ensemble
        return enveloppes

    def iteration_enveloppe(self, marge: float, nombre_iterations: int) -> list[np.ndarray]:
        return self._iteration_enveloppe(marge, nombre_iterations, self.enveloppe_equidiametrique)

    def iteration_enveloppe_cl(self, programme: OpenCLProgram, marge: float, nombre_iterations: int) -> list[np.ndarray]:
        func = lambda m: self.enveloppe_equidiametrique_cl(programme, m)
        return self._iteration_enveloppe(marge, nombre_iterations, func)

    def afficher(self):
        self.ensemble.afficher()

class Calculateur1(CalculateurEnveloppe):
    """ Calculateur de l'enveloppe équidiamétrique utilisant la méthode 1. """
    def __init__(self, ensemble: Ensemble):
        super().__init__(ensemble)

    def enveloppe_equidiametrique(self, marge: float, reduite=True) -> np.ndarray:
        delta = self.ensemble.calc_delta()
        matrice = self.ensemble.matrice_de_travail(marge)
        taille = len(matrice)
        offset = int((taille / 3) * (1 + marge / (2 * marge + 1)))  # Delta est calculée sur la matrice centrale

        for i in range(taille):
            for j in range(taille):
                for (i2, j2, d) in delta:
                    if norme((i, j), (offset + i2, offset + j2)) < d:
                        matrice[i, j] = 1
                        break
        return reduire_matrice(matrice, marge) if reduite else matrice

    def enveloppe_equidiametrique_cl(self, programme: OpenCLProgram, marge: float, reduite=True):
        print("Calcul de la matrice des deltas...")
        delta = self.ensemble.calc_delta_cl_efficace(programme)
        print("Terminé.")
        matrice = self.ensemble.matrice_de_travail(marge)
        taille = len(matrice)
        offset = int((taille / 3) * (1 + marge / (2 * marge + 1)))  # Delta est calculée sur la matrice centrale

        matrice_field = CLField(np.array(matrice, dtype=np.int32))
        matrice_buffer = matrice_field.convert_to_cl(programme.ctx, CLField.mf.READ_WRITE | CLField.mf.USE_HOST_PTR)
        delta_buffer = CLField(np.array(delta, dtype=np.float32)).convert_to_cl(programme.ctx,
                                                                                CLField.mf.READ_ONLY | CLField.mf.USE_HOST_PTR)

        print("Démarrage du calcul de l'enveloppe équidiamétrique...")
        programme.call_function("methode_1", (taille, taille, len(delta)), matrice_buffer, delta_buffer,
                                np.int32(offset),
                                np.int32(taille), np.int32(len(delta)))
        print("Fin du calcul")

        value = matrice_field.retrieve_from_cl(programme.queue, matrice_buffer)
        delta_buffer.release()
        matrice_buffer.release()
        print("Terminé.")

        return reduire_matrice(value, marge) if reduite else value


def evolution(programme: OpenCLProgram, methode_type: type(CalculateurEnveloppe), ensembles: list[Ensemble], marge: float):
    """
    Calcule l'évolution de l'enveloppe équidiamétrique d'une liste d'ensembles de points.
    :param methode_type: Le type de méthode de calcul de l'enveloppe équidiamétrique à utiliser.
    :param programme: L'objet de gestion d'OpenCL.
    :param ensembles: La liste des ensembles de points.
    :param marge: La marge à appliquer à l'ensemble de points pour calculer la matrice de travail.
    :return: La liste des enveloppes équidiamétriques calculées.
    """
    enveloppes = []
    for i in range(len(ensembles)):
        print(f"Iteration {i + 1} / {len(ensembles)}...")
        t = time.time()
        methode = methode_type(ensembles[i])
        enveloppes.append(methode.enveloppe_equidiametrique_cl(programme, marge))
        print(f"Fin de l'itération : {time.time() - t} s")
    return enveloppes
