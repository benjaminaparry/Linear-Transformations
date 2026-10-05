# linear_transformations.py
import time
import numpy as np
from random import random
from matplotlib import pyplot as plt

# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2, n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2, n) ndarray): Transformed matrix
    """
    # stretch the matrix
    stretch_factor = np.array([[a, 0], [0, b]])
    return stretch_factor @ A

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2, n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2, n) ndarray): Transformed matrix
    """
    # shear the matrix
    shear_factor = np.array([[1, a], [b, 1]])
    return shear_factor @ A

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a, b).

    Parameters:
        A ((2, n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2, n) ndarray): Transformed matrix
    """
    # reflect the matrix
    refl_factor = (1 / (a ** 2 + b ** 2)) * np.array([[a ** 2-b ** 2, 2*a*b], [2*a*b, b ** 2 - a ** 2]])
    return refl_factor @ A

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2, n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2, n) ndarray): Transformed matrix
    """
    # rotate the matrix
    rot_factor = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rot_factor @ A


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the Earth and Moon over the time interval [0,T]
    assuming the initial position of the Earth is (x_e,0) and the initial
    position of the Moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The Earth's initial x coordinate.
        x_m (float): The Moon's initial x coordinate.
        omega_e (float): The Earth's angular velocity.
        omega_m (float): The Moon's angular velocity.
    """
    # plot the movement of the sun and moon on a graph
    times = np.linspace(0, T, 1000)
    e_pos = []
    m_pos = []

    for time in times:
        p_e = rotate(np.array([x_e, 0]), time * omega_e)
        mid_vector = rotate(np.array([x_m - x_e, 0]), time * omega_m)
        p_m = mid_vector + p_e
        e_pos.append(p_e)
        m_pos.append(p_m)

    earth = np.array(e_pos)
    moon = np.array(m_pos)

    plt.plot(earth[:, 0], earth[:, 1])
    plt.plot(moon[:, 0], moon[:, 1])
    plt.title("Orbits")
    plt.savefig("orbits.png")


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p)]
                                    for i in range(m)]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    # get times for list matrix multiplication
    domain = 2 ** np.arange(1, 9)
    mmtimes = []
    mvtimes = []
    for n in domain:
        rand1 = random_matrix(n)
        rand2 = random_matrix(n)
        start = time.perf_counter()
        matrix_matrix_product(rand1, rand2)
        mmtimes.append(time.perf_counter() - start)
        randv = random_vector(n)
        start = time.perf_counter()
        matrix_vector_product(rand1, randv)
        mvtimes.append(time.perf_counter() - start)

    # plot the two multiplications
    plt.subplot(122)
    plt.title("Matrix-Matrix Multiplication")
    plt.plot(domain, mmtimes, 'b-')
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.tight_layout()

    plt.subplot(121)
    plt.title("Matrix-Vector Multiplication")
    plt.plot(domain, mvtimes, 'o-')
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.tight_layout()


    plt.savefig("mult_times.png")



# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    # get times for numpy and regular matrix-matrix and matrix-vector multiplications
    domain = 2 ** np.arange(1, 9)
    mmtimes = []
    mvtimes = []
    npmmtimes = []
    npmvtimes = []
    for n in domain:
        rand1 = random_matrix(n)
        rand2 = random_matrix(n)
        start = time.perf_counter()
        matrix_matrix_product(rand1, rand2)
        mmtimes.append(time.perf_counter() - start)
        randv = random_vector(n)
        start = time.perf_counter()
        matrix_vector_product(rand1, randv)
        mvtimes.append(time.perf_counter() - start)
        nprand1 = np.array(rand1)
        nprand2 = np.array(rand2)
        start = time.perf_counter()
        nprand1 @ nprand2
        npmmtimes.append(time.perf_counter() - start)
        nprandv = np.array(randv)
        start = time.perf_counter()
        nprand1 @ nprandv
        npmvtimes.append(time.perf_counter() - start)

    # plot all 4 comparitively on linear and log
    plt.subplot(121)
    plt.plot(domain, mmtimes, 'b-', label="Matrix-Matrix")
    plt.plot(domain, mvtimes, 'o-', label="Matrix-Vector")
    plt.plot(domain, npmmtimes, 'g-', label="Numpy Matrix-Matrix")
    plt.plot(domain, npmvtimes, 'm-', label="Numpy Matrix-Vector")
    plt.title("Linear Execution Times")
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.tight_layout()

    plt.subplot(122)
    plt.loglog(domain, mmtimes, 'b-', base=2, lw=2, label="Matrix-Matrix")
    plt.loglog(domain, mvtimes, 'o-', base=2, lw=2, label="Matrix-Vector")
    plt.loglog(domain, npmmtimes, 'g-', base=2, lw=2, label="Numpy Matrix-Matrix")
    plt.loglog(domain, npmvtimes, 'm-', base=2, lw=2, label="Numpy Matrix-Vector")
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.title("Log Execution Times")
    plt.legend()
    plt.tight_layout()

    plt.savefig("LogandNormalComparisons.png")




def show_horse():
    og_horse = np.load("horse.npy")
    stretch_horse = stretch(og_horse, 1/2, 6/5)
    shear_horse = shear(og_horse, 1/2, 0)
    refl_horse = reflect(og_horse, 0, 1)
    rot_horse = rotate(og_horse, np.pi/2)

    comp_horse = stretch(og_horse, 1/2, 6/5)
    comp_horse = shear(comp_horse, 1/2, 0)
    comp_horse = reflect(comp_horse, 0, 1)
    comp_horse = rotate(comp_horse, np.pi/2)


    plt.subplot(231)
    plt.plot(og_horse[0], og_horse[1], 'k,')
    plt.title("Original")
    plt.tight_layout()

    plt.subplot(232)
    plt.plot(stretch_horse[0], stretch_horse[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.title("Stretch")
    plt.tight_layout()

    plt.subplot(233)
    plt.plot(shear_horse[0], shear_horse[1], 'k,')
    plt.title("Shear")
    plt.tight_layout()

    plt.subplot(234)
    plt.plot(refl_horse[0], refl_horse[1], 'k,')
    plt.title("Reflection")
    plt.tight_layout()

    plt.subplot(235)
    plt.plot(rot_horse[0], rot_horse[1], 'k,')
    plt.title("Rotation")
    plt.tight_layout()

    plt.subplot(236)
    plt.plot(comp_horse[0], comp_horse[1], 'k,')
    plt.title("Composition")
    plt.tight_layout()

    plt.savefig("comparisons.png")

if __name__ == "__main__":
    prob4()
    plt.clf()