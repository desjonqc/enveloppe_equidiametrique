import time

from enveloppe import Ensemble, CalculateurEnveloppe
from opencl import OpenCLProgram


def mesure_delta(ens: Ensemble):
    t = time.time()
    ens.calc_delta()
    print("Temps de calcul de delta :", time.time() - t)

    prog = OpenCLProgram("functions.cl")

    t = time.time()
    ens.calc_delta_cl(prog)
    print("Temps de calcul de delta avec OpenCL :", time.time() - t)

    t = time.time()
    ens.calc_delta_cl_efficace(prog)
    print("Temps de calcul de delta avec OpenCL (efficace) :", time.time() - t)


def mesure_enveloppe(methode: CalculateurEnveloppe):
    prog = OpenCLProgram("functions.cl")

    t = time.time()
    methode.enveloppe_equidiametrique(0.1)
    print("Temps de calcul de l'enveloppe équidiamétrique (s) :", time.time() - t)

    t = time.time()
    methode.enveloppe_equidiametrique_cl(prog, 0.1)
    print("Temps de calcul de l'enveloppe équidiamétrique avec OpenCL (s) :", time.time() - t)
