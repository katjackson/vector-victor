import math

class ShapeError(Exception):
    pass

m = [3, 4]
n = [5, 0]

v = [1, 3, 0]
w = [0, 2, 4]
u = [1, 1, 1]
y = [10, 20, 30]
z = [0, 0, 0]


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
