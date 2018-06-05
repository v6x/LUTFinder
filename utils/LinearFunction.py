from .QuadraticFunction import QuadraticFunction


def get_linearfunction_from_two_point(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    a = (y2 - y1) / (x2 - x1)
    b = (x2 * y1 - x1 * y2) / (x2 - x1)
    return LinearFunction(a, b)


class LinearFunction:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def get_integrate(self):
        return QuadraticFunction(self.a / 2, self.b, 0)