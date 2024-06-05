import numpy as np
from PIL import Image


""" Permet de calculer des formes géométriques simples et de les combiner. """

def importer_image(filename: str) -> np.ndarray:
    image = 1 - np.array(Image.open(filename).convert("L")) / 255
    if len(image) != len(image[0]):
        offset = int((len(image) - len(image[0])) / 2)
        if offset > 0:
            return np.block([np.zeros((len(image), offset)), image, np.zeros((len(image), offset))])
        return np.block([np.zeros((-offset, len(image[0]))), image, np.zeros((-offset, len(image[0])))])
    return image

def ellipse(a: int, b: int, center: tuple, n: int, stroke=1, matrix=None):
    x0, y0 = center
    points = np.zeros((n, n)) if matrix is None else matrix

    x = np.arange(n)
    y = np.arange(n)
    xv, yv = np.meshgrid(x, y)
    mask = (np.abs(((xv - x0)/a) ** 2 + ((yv - y0)/b) ** 2 - 1) <= (2 * stroke / max(a, b)) ** 2)
    points[mask] = 1
    return points

def cercle(radius: int, center: tuple, n: int, stroke=1, matrix=None):
    return ellipse(radius, radius, center, n, stroke, matrix)

def arc(radius: int, center: tuple, angle: float, n: int, stroke=1, matrix=None):
    x0, y0 = center
    points = np.zeros((n, n)) if matrix is None else matrix

    x = np.arange(n)
    y = np.arange(n)
    xv, yv = np.meshgrid(x, y)
    mask = (np.abs(((xv - x0) / radius) ** 2 + ((yv - y0) / radius) ** 2 - 1) <= (2 * stroke / radius) ** 2)
    theta = -np.angle((xv - x0) + 1j * (yv - y0))
    theta[theta < 0] += 2 * np.pi
    mask &= (theta <= angle)
    points[mask] = 1
    return points

def segment(p1: tuple, p2: tuple, n: int, stroke=1, matrix=None):
    x1, y1 = p1
    x2, y2 = p2

    points = np.zeros((n, n)) if matrix is None else matrix

    x = np.arange(n)
    y = np.arange(n)
    xv, yv = np.meshgrid(x, y)
    if x1 == x2:
        mask = (np.abs(xv - x1) < stroke) & (yv >= min(y1, y2)) & (yv <= max(y1, y2))
        points[mask] = 1
        return points

    if y1 == y2:
        mask = (np.abs(yv - y1) < stroke) & (xv >= min(x1, x2)) & (xv <= max(x1, x2))
        points[mask] = 1
        return points

    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1

    mask = (np.abs(yv - k * xv - b) <= stroke) & (xv >= min(x1, x2)) & (xv <= max(x1, x2)) & (yv >= min(y1, y2)) & (yv <= max(y1, y2))
    points[mask] = 1
    return points

def rectangle(side_x: int, side_y: int, center: tuple, n: int, error=2, matrix=None):
    x0, y0 = center
    points = np.zeros((n, n)) if matrix is None else matrix
    for x in range(n):
        for y in range(n):
            # Le carré doit être évidé
            if abs(x - x0) <= side_x / 2 and abs(y - y0) <= side_y / 2 and \
                    not (abs(x - x0) <= side_x / 2 - error and abs(y - y0) <= side_y / 2 - error):
                points[x, y] = 1
    return points

def carre(side: int, center: tuple, n: int, error=2, matrix=None):
    return rectangle(side, side, center, n, error, matrix)


def triangle(a: tuple, b: tuple, c: tuple, n: int, stroke=1, matrix=None):
    T = segment(a, b, n, stroke, matrix)
    T = segment(b, c, n, stroke, T)
    T = segment(c, a, n, stroke, T)
    return T

def triangle_equilateral(a: int, center: tuple, n: int, stroke=1, matrix=None):
    h = np.sqrt(3) / 2 * a
    A = (center[0], center[1] - h / 3)
    B = (center[0] + a / 2, center[1] + 2 * h / 3)
    C = (center[0] - a / 2, center[1] + 2 * h / 3)
    return triangle(A, B, C, n, stroke, matrix)

def triangle_isocele(a: int, b: int, center: tuple, n: int, stroke=1, matrix=None):
    h = np.sqrt(b ** 2 - (a / 2) ** 2)
    A = (center[0], center[1] - h / 3)
    B = (center[0] + a / 2, center[1] + 2 * h / 3)
    C = (center[0] - a / 2, center[1] + 2 * h / 3)
    return triangle(A, B, C, n, stroke, matrix)


def triangle_reuleaux(n, error=0.01):
    a = 1  # side length of the equilateral triangle
    # Calculate the height of the equilateral triangle
    h = np.sqrt(3) / 2 * a

    # Define the vertices of the equilateral triangle
    A = (0, 0)
    B = (a, 0)
    C = (a / 2, h)

    # Create a grid of points
    x = np.linspace(-a / 2, 1.5 * a, n)
    y = np.linspace(-a / 2, 1.5 * a, n)
    xv, yv = np.meshgrid(x, y)

    # Create the arcs of the Reuleaux triangle
    mask0 = (yv >= 0) & (yv <= h) & (xv <= a + error) & (xv >= -error)  # Cut off useless parts of a circle

    mask1 = (np.abs((xv - A[0]) ** 2 + (yv - A[1]) ** 2 - a ** 2) <= error) & mask0
    mask2 = (np.abs((xv - B[0]) ** 2 + (yv - B[1]) ** 2 - a ** 2) <= error) & mask0
    mask3 = (np.abs((xv - C[0]) ** 2 + (yv - C[1]) ** 2 - a ** 2) <= error) & (yv <= 0)

    # Combine the arcs to form the Reuleaux triangle
    return mask1 | mask2 | mask3
