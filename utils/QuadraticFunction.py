class QuadraticFunction:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def calc(self, x):
        return self.a * x * x + self.b * x + self.c

    def vertical_move(self, point):
        """
        move function vertically so that the point is on this function
        :param point: point (x, y)
        :return:
        """
        x, y = point
        self.c = y - self.a * x * x - self.b * x