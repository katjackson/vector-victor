import math
from Vector2d import *
from nose.tools import raises


def are_equal(x, y, tolerance=0.001):
    """Helper function to compare floats, which are often not quite equal."""
    return abs(x - y) <= tolerance


m = [3, 4]
n = [5, 0]

v = [1, 3, 0]
w = [0, 2, 4]
u = [1, 1, 1]
y = [10, 20, 30]
z = [0, 0, 0]

class TestVector2d():

    def test_shape_vectors(self):
        """shape takes a vector or matrix and return a tuple with the
        number of rows (for a vector) or the number of rows and columns
        (for a matrix.)"""
        assert Vector2d.shape(self, [1]) == (1,)
        assert Vector2d.shape(self, m) == (2,)
        assert Vector2d.shape(self, v) == (3,)


    def test_vector_add(self):
        """
        [a b]  + [c d]  = [a+c b+d]
        Matrix + Matrix = Matrix
        """
        assert Vector2d.vector_add(Vector2d, v, w) == [1, 5, 4]
        assert Vector2d.vector_add(Vector2d, u, y) == [11, 21, 31]
        assert Vector2d.vector_add(Vector2d, u, z) == u


    def test_vector_add_is_commutative(self):
        assert Vector2d.vector_add(Vector2d, w, y) == Vector2d.vector_add(Vector2d, y, w)


    @raises(ShapeError)
    def test_vector_add_checks_shapes(self):
        """Shape rule: the vectors must be the same size."""
        Vector2d.vector_add(Vector2d, m, v)


    def test_vector_sub(self):
        """
        [a b]  - [c d]  = [a-c b-d]
        Matrix + Matrix = Matrix
        """
        assert Vector2d.vector_sub(Vector2d, v, w) == [1, 1, -4]
        assert Vector2d.vector_sub(Vector2d, w, v) == [-1, -1, 4]
        assert Vector2d.vector_sub(Vector2d, y, z) == y
        assert Vector2d.vector_sub(Vector2d, w, u) == Vector2d.vector_sub(Vector2d, z, Vector2d.vector_sub(Vector2d, u, w))


    @raises(ShapeError)
    def test_vector_sub_checks_shapes(self):
        """Shape rule: the vectors must be the same size."""
        Vector2d.vector_sub(Vector2d, m, v)


    # def test_vector_sum(self):
    #     """vector_sum can take any number of vectors and add them together."""
    #     assert Vector2d.vector_sum(v, w, u, y, z) == [12, 26, 35]
    #
    #
    # @raises(ShapeError)
    # def test_vector_sum_checks_shapes(self):
    #     """Shape rule: the vectors must be the same size."""
    #     Vector2d.vector_sum(Vector2d, v, w, m, y)

        #
    # def test_dot():
    #     """
    #     dot([a b], [c d])   = a * c + b * d
    #     dot(Vector, Vector) = Scalar
    #     """
    #     assert dot(w, y) == 160
    #     assert dot(m, n) == 15
    #     assert dot(u, z) == 0
    #
    #
    # @raises(ShapeError)
    # def test_dot_checks_shapes():
    #     """Shape rule: the vectors must be the same size."""
    #     dot(v, m)
    #
    #
    # def test_vector_multiply():
    #     """
    #     [a b]  *  Z     = [a*Z b*Z]
    #     Vector * Scalar = Vector
    #     """
    #     assert vector_multiply(v, 0.5) == [0.5, 1.5, 0]
    #     assert vector_multiply(m, 2) == [6, 8]
    #
    #
    # def test_vector_mean():
    #     """
    #     mean([a b], [c d]) = [mean(a, c) mean(b, d)]
    #     mean(Vector)       = Vector
    #     """
    #     assert vector_mean(m, n) == [4, 2]
    #     assert vector_mean(v, w) == [0.5, 2.5, 2]
    #     assert are_equal(vector_mean(v, w, u)[0], 2 / 3)
    #     assert are_equal(vector_mean(v, w, u)[1], 2)
    #     assert are_equal(vector_mean(v, w, u)[2], 5 / 3)
    #
    #
    # def test_magnitude():
    #     """
    #     magnitude([a b])  = sqrt(a^2 + b^2)
    #     magnitude(Vector) = Scalar
    #     """
    #     assert magnitude(m) == 5
    #     assert magnitude(v) == math.sqrt(10)
    #     assert magnitude(y) == math.sqrt(1400)
    #     assert magnitude(z) == 0
    #
    #
    # A = [[1, 0, 0],
    #      [0, 1, 0],
    #      [0, 0, 1]]
    # B = [[1, 2, 3],
    #      [4, 5, 6],
    #      [7, 8, 9]]
    # C = [[1, 2],
    #      [2, 1],
    #      [1, 2]]
    # D = [[1, 2, 3],
    #      [3, 2, 1]]
    #
    # #
    # # ADVANCED MODE TESTS BELOW
    # # UNCOMMENT THEM FOR ADVANCED MODE!
    #
    # def test_shape_matrices():
    #     """shape takes a vector or matrix and return a tuple with the
    #     number of rows (for a vector) or the number of rows and columns
    #     (for a matrix.)"""
    #     assert shape(A) == (3, 3)
    #     assert shape(C) == (3, 2)
    #     assert shape(D) == (2, 3)
    #
    #
    # def test_matrix_row():
    #     """
    #            0 1  <- rows
    #        0 [[a b]]
    #        1 [[c d]]
    #        ^
    #      columns
    #     """
    #     assert matrix_row(A, 0) == [1, 0, 0]
    #     assert matrix_row(B, 1) == [4, 5, 6]
    #     assert matrix_row(C, 2) == [1, 2]
    #
    #
    # def test_matrix_col():
    #     """
    #            0 1  <- rows
    #        0 [[a b]]
    #        1 [[c d]]
    #        ^
    #      columns
    #     """
    #     assert matrix_col(A, 0) == [1, 0, 0]
    #     assert matrix_col(B, 1) == [2, 5, 8]
    #     assert matrix_col(D, 2) == [3, 1]
    #
    #
    # def test_matrix_matrix_add():
    #     assert matrix_add(A, B) == [[2, 2, 3],
    #                                 [4, 6, 6],
    #                                 [7, 8, 10]]
    #
    #
    # @raises(ShapeError)
    # def test_matrix_add_checks_shapes():
    #     """Shape rule: the rows and columns of the matrices must be the same size."""
    #     matrix_add(C, D)
    #
    #
    # def test_matrix_matrix_sub():
    #     assert matrix_sub(A, B) == [[ 0, -2, -3],
    #                                 [-4, -4, -6],
    #                                 [-7, -8, -8]]
    #
    #
    # @raises(ShapeError)
    # def test_matrix_sub_checks_shapes():
    #     """Shape rule: the rows and columns of the matrices must be the same size."""
    #     matrix_sub(C, D)
    #
    #
    # def test_matrix_scalar_multiply():
    #     """
    #     [[a b]   *  Z   =   [[a*Z b*Z]
    #      [c d]]              [c*Z d*Z]]
    #
    #     Matrix * Scalar = Matrix
    #     """
    #     assert matrix_scalar_multiply(C, 3) == [[3, 6],
    #                                             [6, 3],
    #                                             [3, 6]]
    #     assert matrix_scalar_multiply(B, 2) == [[ 2,  4,  6],
    #                                             [ 8, 10, 12],
    #                                             [14, 16, 18]]
    #
    #
    # def test_matrix_vector_multiply():
    #     """
    #     [[a b]   *  [x   =   [a*x+b*y
    #      [c d]       y]       c*x+d*y
    #      [e f]                e*x+f*y]
    #
    #     Matrix * Vector = Vector
    #     """
    #     assert matrix_vector_multiply(A, [2, 5, 4]) == [2, 5, 4]
    #     assert matrix_vector_multiply(B, [1, 2, 3]) == [14, 32, 50]
    #     assert matrix_vector_multiply(C, [3, 4]) == [11, 10, 11]
    #     assert matrix_vector_multiply(D, [0, 1, 2]) == [8, 4]
    #
    #
    # @raises(ShapeError)
    # def test_matrix_vector_multiply_checks_shapes():
    #     """Shape Rule: The number of rows of the vector must equal the number of
    #     columns of the matrix."""
    #     matrix_vector_multiply(C, [1, 2, 3])
    #
    #
    # def test_matrix_matrix_multiply():
    #     """
    #     [[a b]   *  [[w x]   =   [[a*w+b*y a*x+b*z]
    #      [c d]       [y z]]       [c*w+d*y c*x+d*z]
    #      [e f]                    [e*w+f*y e*x+f*z]]
    #
    #     Matrix * Matrix = Matrix
    #     """
    #     assert matrix_matrix_multiply(A, B) == B
    #     assert matrix_matrix_multiply(B, C) == [[8, 10],
    #                                             [20, 25],
    #                                             [32, 40]]
    #     assert matrix_matrix_multiply(C, D) == [[7, 6, 5],
    #                                             [5, 6, 7],
    #                                             [7, 6, 5]]
    #     assert matrix_matrix_multiply(D, C) == [[8, 10], [8, 10]]
    #
    #
    # @raises(ShapeError)
    # def test_matrix_matrix_multiply_checks_shapes():
    #     """Shape Rule: The number of columns of the first matrix must equal the
    #     number of rows of the second matrix."""
    #     matrix_matrix_multiply(A, D)
