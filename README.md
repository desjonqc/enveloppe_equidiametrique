# Enveloppe équidiamétrique d'un ensemble connexe borné

## Définition

Soient $E$ un ensemble et $D$ un sous-ensemble connexe et borné de E. On définit l'enveloppe équidiamétrique de $D$ par

E={M∈E ∣ ∃ (A,B)∈D2, δ(M,A)≤δ(A,B)}où $\delta$ représente la distance euclidienne.

## Présentation des objets

### Ensemble

La simulation informatique nécessite la discrétisation des ensembles considérés. L'hypothèse *borné* se traduit donc simplement par *fini*.

Un ensemble $D$ est représenté par l'objet `enveloppe.Ensemble`. Pour créer un ensemble à partir d'un tableau numpy d'éléments on utilise la synthaxe :

```python
from enveloppe import Ensemble

elem = ...  # Tableau numpy contenant les éléments
ensemble = Ensemble(elem)
```
### Méthode de calcul

Pour créer une méthode de calcul, il suffit de créer un objet héritant de la classe `enveloppe.CalculateurEnveloppe` et de remplir comme on le souhaite les fonctions abstraites.


Pour utiliser la seule méthode implémentée à partir d'un ensemble $D$, il suffit d'écrire :

```python
from enveloppe import Calculateur1

methode = Calculateur1(ensemble)
```
### Calcul de l'enveloppe

**Pour tout usage de OpenCL, écrire au préalable :**

```python
from opencl import OpenCLProgram

programme = OpenCLProgram("functions.cl")
```
Pour calculer l'enveloppe équidiamétrique, on utilise la synthaxe suivante :

```python
resultat = methode.enveloppe_equidiametrique_cl(programme, marge, reduite)  # Calcul d'une enveloppe

resultats = methode.iteration_enveloppe_cl(programme, marge, N)  # Calcul de N itérations (renvoie une liste contenant les N enveloppes)
```
Pour calculer les enveloppes d'un ensemble $D$ évolutif, afin d'observer le comportement de l'enveloppe d'un ensemble D variable on utilise la fonction `enveloppe.evolution(programme, methode_classe, ensembles, marge)` où methode_classe est le type (au sens python) du calculateur à appliquer, ensembles est une liste représentant l'évolution de D.

Par exemple,

```python
from enveloppe import evolution

ensembles = [ensemble1, ensemble2, ...]  # L'évolution de D
evolutions = evolution(programme, Calculateur1, ensembles, marge)
```
## Affichage des résultats

Pour afficher simplement un ensemble, on utilise la fonction `Ensemble.afficher()`.

### Afficher les itérations

Pour afficher les itérations d'enveloppes, on peut utiliser l'objet `affichage.ResultatAnimation` pour afficher au court du temps les différentes itérations. Par exemple :

```python
from affichage import ResultatAnimation

animation = ResultatAnimation(ensemble, resultats, interval)
animation.afficher()  # Affiche l'animation

animation.sauvegarder("chemin_vers_le_fichier.gif")  # Sauvegarde l'animation
```
### Afficher les évolutions

Même principe, mais l'objet diffère : on utilise l'objet `affichage.EvolutionAnimation`.

```python
from affichage import EvolutionAnimation

animation = EvolutionAnimation(evolutions, ensembles, interval)
animation.afficher()  # Affiche l'animation

animation.sauvegarder("chemin_vers_le_fichier.gif")  # Sauvegarde l'animation
```
Par exemple, en prenant $D$ égal à un arc de cercle de degré $\theta$ variant de $0$ à $2\pi$, on obtient ce résultat :

<img src="file://C:/Cours_Java/evolution_arcs.gif" width="400" />


## Création de formes géométriques

Il y a plusieurs fonctions à disposition dans le module `geometrie` permettant de créer des formes simples et de les combiner. Elles renvoient toutes un tableau numpy correspondant à la frontière de l'ensemble représentant la forme géométrique désirée, sous le bon format pour être utilisé sans transformation supplémentaire dans l'algorithme ci-dessus.

<ul>

<li><b>importer_image</b> importe une image numérique en <b>noir et blanc</b>.</li>
<li><b>ellipse</b></li>
<li><b>circle</b></li>
<li><b>arc</b></li>
<li><b>segment</b></li>
<li><b>rectangle</b></li>
<li><b>carre</b></li>
<li><b>triangle</b></li>
<li><b>triangle_isocele</b></li>
<li><b>triangle_equilateral</b></li>
<li><b>triangle_reuleaux</b></li>
</ul>

# Conclusion

On obtient un algorithme permettant de réaliser une approximation numérique de l'enveloppe équidiamétrique d'une ensemble convexe borné.

<img src="file://C:/Cours_Java/texas.png" width="500" >
