import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T
S_MATRIX = [[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]]


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def get_random_points():
    """
    Task 11
    Using the identity matrix as the covariance matrix to generate random points and then
    plot them
    :return:
    """

    plot_3d(x_y_z)
    plt.title("Q11 : Random 3D points using the identity matrix")
    # plt.savefig("Q11.pdf")
    plt.show()

# get_random_points()


def transform_the_data():
    """
    Task 12
    Transform the data with the given matrix
    """

    new_matrix = np.dot(S_MATRIX, x_y_z)
    plot_3d(new_matrix)
    plt.title("Q12 : Transform random 3D points with the covariance matrix named S")
    # plt.savefig("Q12.pdf")
    plt.show()

    # Numerically
    print(np.cov(new_matrix))

    print("\n")

    # Analytically
    print(np.dot(S_MATRIX, np.transpose(S_MATRIX)))

# transform_the_data()


def multiply_orthogonal():
    """
    Task 13
    :return:
    """

    matrix_orthogonal = get_orthogonal_matrix(3)

    new_matrix = np.dot(matrix_orthogonal, S_MATRIX)
    print(np.dot(new_matrix, new_matrix.T))

    plot_3d(np.dot(matrix_orthogonal, np.dot(S_MATRIX, x_y_z)))
    plt.title("Q13 : Multiply by orthogonal matrix")
    # plt.savefig("Q13.pdf")
    plt.show()

# multiply_orthogonal()

def marginal_gaussian():
    """
    Task 14
    :return:
    """

    matrix_orthogonal = get_orthogonal_matrix(3)
    plot_2d(np.dot(matrix_orthogonal, np.dot(S_MATRIX, x_y_z)))
    plt.title("Q14 : Marginal gaussian")
    # plt.savefig("Q14.pdf")
    plt.show()

# marginal_gaussian()


def distribution_gaussian():
    """
    Task 15
    :return:
    """

    good_data = np.where((-0.4 < x_y_z[2]) & (x_y_z[2] < 0.1))
    selection_final = x_y_z[0 : 2, good_data]
    plot_2d(selection_final)
    plt.title("Q15 : Distributin Gaussian")
    # plt.savefig("Q15.pdf")
    plt.show()

# distribution_gaussian()