import math

class ShapeError(Exception):
    pass


def shape(num_set):
    rows = len(num_set)
    try:
        columns = len(num_set[0])
        return (rows, columns)
    except:
        return (rows, )


def vector_add(vector_a, vector_b):
    if shape(vector_a) != shape(vector_b):
        raise ShapeError(Exception)

    return [a + b for a, b in zip(vector_a, vector_b)]


def vector_sub(vector_a, vector_b):
    if shape(vector_a) != shape(vector_b):
        raise ShapeError(Exception)

    return [a - b for a, b in zip(vector_a, vector_b)]


def vector_sum(*args):
    shapes = {shape(arg) for arg in args}
    if len(shapes) > 1:
        raise ShapeError(Exception)

    return [sum(x) for x in zip(*args)]


def dot(vector_a, vector_b):
    if shape(vector_a) != shape(vector_b):
        raise ShapeError(Exception)

    multiplied_points = [(a * b) for a, b in zip(vector_a, vector_b)]
    return sum(a for a in multiplied_points)


def vector_multiply(vector, scalar):
    return [a * scalar for a in vector]


def vector_mean(*args):
    list_of_args = [arg for arg in args]
    return vector_multiply(vector_sum(*args), (1 / len(list_of_args)))


def magnitude(vector):
    vector_squared_scalar = dot(vector, vector)
    return math.sqrt(vector_squared_scalar)


def matrix_row(matrix, row):
    return matrix[row]


def matrix_col(matrix, col):
    return [x[col] for x in matrix]


def matrix_add(matrix_a, matrix_b):
    if shape(matrix_a) != shape(matrix_b):
        raise ShapeError(Exception)

    return [vector_add(a, b) for a, b in zip(matrix_a, matrix_b)]


def matrix_sub(matrix_a, matrix_b):
    if shape(matrix_a) != shape(matrix_b):
        raise ShapeError(Exception)

    return [vector_sub(a, b) for a, b in zip(matrix_a, matrix_b)]


def matrix_scalar_multiply(matrix, scalar):
    return [vector_multiply(vector, scalar) for vector in matrix]


def matrix_vector_multiply(matrix, vector):
    matrix_rows, matrix_columns = shape(matrix)
    vector_rows = shape(vector)
    if vector_rows != (matrix_columns,):
        raise ShapeError(Exception)

    return [dot(row, vector) for row in matrix]


def matrix_matrix_multiply(matrix_a, matrix_b):
    matrix_a_rows, matrix_a_col = shape(matrix_a)
    matrix_b_rows, matrix_b_col = shape(matrix_b)
    if matrix_a_col != matrix_b_rows:
        raise ShapeError(Exception)

    new_matrix = [[dot(row, matrix_col(matrix_b, column)) for column in range(matrix_b_col)] for row in matrix_a]
    return new_matrix
