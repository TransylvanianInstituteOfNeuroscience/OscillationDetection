import math

import numpy as np
PRECISION = 1e-8    # Arbitrary zero for real-world purposes

def plane_from_points(points):
    # The adjusted plane crosses the centroid of the point collection
    centroid = np.mean(points, axis=0)

    # Use SVD to calculate the principal axes of the point collection
    # (eigenvectors) and their relative size (eigenvalues)
    _, values, vectors = np.linalg.svd(points - centroid)

    # Each singular value is paired with its vector and they are sorted from
    # largest to smallest value.
    # The adjusted plane plane must contain the eigenvectors corresponding to
    # the two largest eigenvalues. If only one eigenvector is different
    # from zero, then points are aligned and they don't define a plane.
    if values[1] < PRECISION:
        raise ValueError("Points are aligned, can't define a plane")

    # So the plane normal is the eigenvector with the smallest eigenvalue
    normal = vectors[2]

    # Calculate the coefficients (a,b,c,d) of the plane's equation ax+by+cz+d=0.
    # The first three coefficients are given by the normal, and the fourth
    # one (d) is the plane's signed distance to the origin of coordinates
    d = -np.dot(centroid, normal)
    plane = np.append(normal, d)

    # If the smallest eigenvector is close to zero, the collection of
    # points is perfectly flat. The larger the eigenvector, the less flat.
    # You may wish to know this.
    thickness = values[2]

    return plane, thickness

def point_to_plane_distance(point, plane):
    return abs(plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3]) / math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)


def line_from_two_points(pointA, pointB):
    return (pointA[1] - pointB[1], pointB[0] - pointA[0], pointA[0]*pointB[1] - pointA[1]*pointB[0])


def point_to_line_distance(point, line):
    return abs(line[0] * point[0] + line[1] * point[1] + line[2]) / math.sqrt(line[0] ** 2 + line[1] ** 2)


def plane_from_three_points(pointA, pointB, pointC):
    a1 = pointB[0] - pointA[0]
    b1 = pointB[1] - pointA[1]
    c1 = pointB[2] - pointA[2]
    a2 = pointC[0] - pointA[0]
    b2 = pointC[1] - pointA[1]
    c2 = pointC[2] - pointA[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * pointA[0] - b * pointA[1] - c * pointA[2])

    return (a,b,c,d)


def perpendicular_plane_from_plane_and_two_points(plane, pointA, pointB):
    # Find direction vector of points A and B
    a = pointB[0] - pointA[0]
    b = pointB[1] - pointA[1]
    c = pointB[2] - pointA[2]

    # Values that are calculated
    # and simplified from the
    # cross product
    A = (b * plane[2] - c * plane[1])
    B = (a * plane[2] - c * plane[0])
    C = (a * plane[1] - b * plane[0])
    D = -(A * plane[0] - B * plane[1] + C * plane[2])

    return [A, B, C, D]
