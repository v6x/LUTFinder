import math


def interpolate(values, distances):
    sum = 0
    for distance in distances:
        sum += distance
    portions = []
    for distance in distances:
        portions.append(distance / sum)
    ans = 0
    for idx, value in enumerate(values):
        ans += portions[idx] * value
    return ans


def interpolate_from_ndarray(nd_values, coord, simple_factor):
    near_coords = find_nearest_simplecoords_3d(coord, simple_factor)
    distances = get_distances(coord, near_coords)
    values = []
    for near_coord in near_coords:
        x, y, z = near_coord
        values.append(nd_values.item((x, y, z)))
    return interpolate(values, distances)


def find_nearest_simplecoords_3d(coord, simple_factor):
    x, y, z = coord
    base_x = x - x % simple_factor
    base_y = y - y % simple_factor
    base_z = z - z % simple_factor
    return (base_x, base_y, base_z), (base_x, base_y, base_z + simple_factor), (base_x, base_y + simple_factor, base_z), (base_x, base_y + simple_factor, base_z + simple_factor)\
        , (base_x + simple_factor, base_y, base_z), (base_x + simple_factor, base_y, base_z + simple_factor), (base_x + simple_factor, base_y + simple_factor, base_z), (base_x + simple_factor, base_y + simple_factor, base_z + simple_factor)


def get_distances(coord, near_coords):
    distances = []
    for near_coord in near_coords:
        distances.append(distance(coord, near_coord))
    return distances


def distance(coord1, coord2):
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2

    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))