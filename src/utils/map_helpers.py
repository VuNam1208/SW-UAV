import copy
import math, random
import sys
from pathlib import Path
from tkinter import filedialog

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from .calculation_helpers import *

parent_dir = Path(__file__).parent.parent


# -------------User defined functions------------
def area_of_polygon(vertices):
    """Calculate the area of a polygon defined by the given vertices.

    Args:
        vertices (List[List(float)]): [[x1, y1], [x2, y2], ...] such that the polygon is defined by the vertices.

    Returns:
        Float: Area of the polygon.
    """
    area = 0
    n = len(vertices)
    if len(vertices) < 3:
        return 0

    for i in range(n - 1):
        p1 = vertices[i]
        p2 = vertices[i + 1]
        area += convert_degrees_to_radius(p2[1] - p1[1]) * (
            2
            + math.sin(convert_degrees_to_radius(p1[0]))
            + math.sin(convert_degrees_to_radius(p2[0]))
        )
    area = area * EARTH_RADIUS**2 / 2

    # return in square meters, kilometers, and ha
    return {
        "m2": abs(area),
        "km2": abs(area) / 1e6,
        "ha": abs(area) / 1e4,
    }


def split_polygon_into_areas(vertices, number_of_parts):
    """Split a polygon defined by the given vertices into a number of parts.

    Args:
        vertices (List[List(float)]): [[x1, y1], [x2, y2], ...] such that the polygon is defined by the vertices.
        number_of_parts (int): Number of parts to split the polygon into.

    Returns:
        List[List[List(float)]]: A list of lists of vertices defining the split areas.
    """
    splitted_areas = {}
    result = {}
    # Convert the list of positions to a NumPy array
    points = np.array(vertices)

    ref_lat = min(points, key=lambda x: x[0])[0]
    ref_lon = min(points, key=lambda x: x[1])[1]

    if len(points) < 3:
        return result

    cartesian_coordinates = convert_to_cartesian(points)

    if not is_polygon_convex(cartesian_coordinates):
        return result

    # 1. Find longest edge
    longest_edge_points = find_longest_edge(cartesian_coordinates)
    # 2. Find the perpendicular line to the longest edge at midpoint
    main_perpendicular_line = perpendicular_lines_at_points(longest_edge_points, N=2)[0]
    intersection_pts_of_perpendicular_line = find_polygon_line_intersections(
        cartesian_polygon=cartesian_coordinates, line=main_perpendicular_line
    )
    perp_line_within_polygon = intersection_pts_of_perpendicular_line
    # 3. Divide the perpendicular line into equal parts, and find the lines parallel to the longest edge
    parallel_lines_to_longest_edge = perpendicular_lines_at_points(
        perp_line_within_polygon, N=number_of_parts
    )
    parallel_lines_to_longest_edge.sort(key=lambda x: x[1])  # sort by the intercept

    # 4. Find the intersection points of the parallel lines with the polygon
    intersection_pts_of_parallel_lines = [
        find_polygon_line_intersections(cartesian_coordinates, line)
        for line in parallel_lines_to_longest_edge
    ]

    # 5. Split the polygon into equal parts
    for index, line in enumerate(parallel_lines_to_longest_edge):
        for point in cartesian_coordinates[:-1]:
            if is_left_of_line(point, line):
                splitted_areas.setdefault(index, {}).setdefault("left", []).append(point)
            else:
                splitted_areas.setdefault(index, {}).setdefault("right", []).append(point)

    result[0] = splitted_areas[0]["right"] + intersection_pts_of_parallel_lines[0]

    # between 2 lines add the points that are between them
    for i in range(1, number_of_parts - 1):
        addition_pts = [
            pt
            for pt in cartesian_coordinates[:-1]
            if is_between_lines(
                pt, parallel_lines_to_longest_edge[i - 1], parallel_lines_to_longest_edge[i]
            )
        ]
        # addition_pts = [pt for pt in splitted_areas[i]['right'] if pt not in result[i - 1]]
        result[i] = (
            addition_pts
            + intersection_pts_of_parallel_lines[i - 1]
            + intersection_pts_of_parallel_lines[i]
        )

    result[number_of_parts - 1] = (
        splitted_areas[number_of_parts - 1 - 1]["left"] + intersection_pts_of_parallel_lines[-1]
    )
    # convert the result to lat lon
    gps_result = {}
    for key, value in result.items():
        gps_result[key] = []
        for point in value:
            gps_result[key].append(convert_to_lat_lon([ref_lat, ref_lon], point))

    return {
        "cartesian": result,
        "lat_lon": gps_result,
    }


def split_polygon_into_areas_old(vertices, number_of_parts):
    """Split a polygon defined by the given vertices into a number of parts.

    Args:
        vertices (List[List(float)]): [[x1, y1], [x2, y2], ...] such that the polygon is defined by the vertices.
        number_of_parts (int): Number of parts to split the polygon into.

    Returns:
        List[List[List(float)]]: A list of lists of vertices defining the split areas.
    """
    # global angle, midpoint, min_lat, min_lon
    positions = vertices
    number_of_part = number_of_parts

    min_lat = min(positions, key=lambda x: x[0])[0]
    min_lon = min(positions, key=lambda x: x[1])[1]
    # Convert the geographic positions to Cartesian coordinates
    cartesian_coordinates = convert_to_cartesian(positions)
    # print("Cartesian Coordinates:")
    # for coord in cartesian_coordinates:
    #     print(coord)

    # Find the largest edge
    _, longest_edge_point = find_longest_edge(cartesian_coordinates)
    # print(f"\nEdge: {longest}")
    # for coord in longest_edge_point:
    #     print(coord)

    # Find midpont of largest edge
    midpoint = find_midpoint(longest_edge_point[0], longest_edge_point[1])
    # print(f"\nMidpoint: {midpoint}")
    new = calculate_new_lat_lon(min_lat, min_lon, midpoint[1], midpoint[0])
    # print(f"\nGPS Midpoint: {new}")

    # Find line equation of largest edge (to figure out the slope of it)
    slope, intercept = line_equation_from_points(longest_edge_point[0], longest_edge_point[1])
    # print(f"\nSlope: {slope}")
    angle = angle_with_x_axis(slope)
    # print(f"\nAngle: {angle}")

    new_point = rotate_and_shift_point(
        midpoint[0],
        midpoint[1],
        (-angle),
        midpoint[0],
        midpoint[1],
        (-midpoint[0]),
        (-midpoint[1]),
    )
    # print(f"ROTATED MIDPOINT{new_point}")

    # Find the perpendicular's line equation (perpendicular of largest edge)
    perp_slope, perp_intercept = perpendicular_line_equation(midpoint, slope)
    # print(f"\nPerp_slope: {perp_slope},Perp_intercep: {perp_intercept}")

    # Find the other intersection of the perpendicular with the polygon
    intersect_point = does_line_intersect_polygon(
        midpoint, perp_slope, perp_intercept, cartesian_coordinates
    )
    # print(f"\nIntersect: {intersect_point[0]},{intersect_point[1]}")
    new = calculate_new_lat_lon(min_lat, min_lon, intersect_point[1], intersect_point[0])
    # print(f"\nGPS Intersect: {new}")

    # ---------------------------------

    # ---------------------------------
    # Divide the perpendicular into equal parts
    perpendicular_points = divide_line_into_segments(
        midpoint[0], midpoint[1], intersect_point[0], intersect_point[1], number_of_part
    )
    # print(f"\nPerpendicular Points: {perpendicular_points}")
    per_GPS_list = []
    for point in perpendicular_points:
        new = calculate_new_lat_lon(min_lat, min_lon, point[1], point[0])
        per_GPS_list.append(new)
        # print(f"{point}")

    # Find the divide point on the polygon edge
    div_GPS_list = []
    div_points = divide_points(perpendicular_points, cartesian_coordinates, perp_slope, slope)
    for point in div_points:
        new = calculate_new_lat_lon(min_lat, min_lon, point[1], point[0])
        div_GPS_list.append(new)
        # print(f"{new}")

    # Rotate and shift the coordinate
    rotated_div_points = []
    # print(f"DIV_ROTATED")
    for point in div_points:
        new_point = rotate_and_shift_point(
            point[0], point[1], (-angle), midpoint[0], midpoint[1], (-midpoint[0]), (-midpoint[1])
        )
        rotated_div_points.append(new_point)
        # print(f"{new_point}")

    rotated_perpendicular_points = []
    # print(f"PERP_ROTATED")
    for point in perpendicular_points:
        new_point = rotate_and_shift_point(
            point[0], point[1], (-angle), midpoint[0], midpoint[1], (-midpoint[0]), (-midpoint[1])
        )
        rotated_perpendicular_points.append(new_point)
        # print(f"{new_point}")

    # print(f"PERP_UNROTATED")
    # for point in perpendicular_points:
    # print(f"{point}")

    # print(f"POLYGON_ROTATED")
    rotated_cartesian_coordinates = []
    for point in cartesian_coordinates:
        new_point = rotate_and_shift_point(
            point[0], point[1], (-angle), midpoint[0], midpoint[1], (-midpoint[0]), (-midpoint[1])
        )
        rotated_cartesian_coordinates.append(new_point)
        # print(f"{new_point}")
    # print(f"POLYGON_UNROTATED")
    # for point in cartesian_coordinates:
    # print(f"{point}")

    rotate_polygon = []
    # Points lie on polygon egde = vertices + divide points
    rotate_polygon = rotated_div_points + rotated_cartesian_coordinates

    # Separate the point into different parts
    rotated_area = split_area(rotate_polygon, rotated_perpendicular_points)
    final_area = []

    # print(f"POLYGON_UNROTATED_BACK")
    # for point in rotated_cartesian_coordinates:
    #     # convert back in previous coordinate
    #     new_point = revert_rotate_and_shift_point(
    #         point[0],
    #         point[1],
    #         (-angle),
    #         midpoint[0],
    #         midpoint[1],
    #         (-midpoint[0]),
    #         (-midpoint[1]),
    #         clockwise=True,
    #     )
    # print(f"{new_point}")

    for i in range(len(rotated_area)):
        area = rotated_area[i]
        unrotated_area = []
        # print(f"{area}")
        for point in area:
            # convert back in previous coordinate
            new_point = revert_rotate_and_shift_point(
                point[0],
                point[1],
                (-angle),
                midpoint[0],
                midpoint[1],
                (-midpoint[0]),
                (-midpoint[1]),
                clockwise=True,
            )
            unrotated_area.append(new_point)
        per_GPS_list = []
        for point in unrotated_area:
            new = calculate_new_lat_lon(min_lat, min_lon, point[1], point[0])
            per_GPS_list.append(new)
            # print(f"{new}")
        # Convert the list of positions to a NumPy array

        points = np.array(per_GPS_list)

        # Calculate the convex hull
        hull = ConvexHull(points)

        # Extract the vertices of the convex hull
        hull_vertices = points[hull.vertices]

        # Convert the vertices back to a list of tuples
        points = [tuple(point) for point in hull_vertices]
        final_area.append(points)

    return final_area, rotated_area, angle, midpoint, min_lat, min_lon


def split_grids(rotated_area, angle, midpoint, min_lat, min_lon, grid_size, n_areas):

    #HaoNV35 Start.
    distance = 10
    #HaoNV35 End.

    if n_areas in (0, 1):
        area = rotated_area[0]
        min_lat = min(area, key=lambda x: x[0])[0]
        min_lon = min(area, key=lambda x: x[1])[1]
        # Convert the geographic positions to Cartesian coordinates
        cartesian_coordinates = convert_to_cartesian(area)
        print("Cartesian Coordinates:")
        for coord in cartesian_coordinates:
            print(coord)

        # Find the largest edge
        longest, longest_edge_point = find_longest_edge(cartesian_coordinates)
        print(f"\nEdge: {longest}")
        for coord in longest_edge_point:
            print(coord)

        # Find midpont of largest edge
        midpoint = find_midpoint(longest_edge_point[0], longest_edge_point[1])
        print(f"\nMidpoint: {midpoint}")
        new = calculate_new_lat_lon(min_lat, min_lon, midpoint[1], midpoint[0])
        print(f"\nGPS Midpoint: {new}")

        # Find line equation of largest edge (to figure out the slope of it)
        slope, intercept = line_equation_from_points(longest_edge_point[0], longest_edge_point[1])
        print(f"\nSlope: {slope}")
        angle = angle_with_x_axis(slope)
        print(f"\nAngle: {angle}")

        rotated = []
        for point in cartesian_coordinates:
            new_point = rotate_and_shift_point(
                point[0],
                point[1],
                (-angle),
                midpoint[0],
                midpoint[1],
                (-midpoint[0]),
                (-midpoint[1]),
            )
            rotated.append(new_point)
            print(f"{new_point}")

        points = np.array(rotated)

        # Calculate the convex hull
        hull = ConvexHull(points)

        # Extract the vertices of the convex hull
        hull_vertices = points[hull.vertices]

        # Convert the vertices back to a list of tuples
        points = [tuple(point) for point in hull_vertices]

        #HaoNV35 Start.
        grid_points = generate_grid(points, int(distance))
        #HaoNV35 End.

        unrotated_area = []
        for point in grid_points:
            # convert back in previous coordinate
            new_point = revert_rotate_and_shift_point(
                point[0],
                point[1],
                (-angle),
                midpoint[0],
                midpoint[1],
                (-midpoint[0]),
                (-midpoint[1]),
                clockwise=True,
            )
            unrotated_area.append(new_point)
        per_GPS_list = []
        for point in unrotated_area:
            new = calculate_new_lat_lon(min_lat, min_lon, point[1], point[0])
            per_GPS_list.append(new)

        return per_GPS_list
    else:
        # n areas > 1
        areas_dot = []
        for i, area in enumerate(rotated_area):

            points = np.array(area)

            # Calculate the convex hull
            hull = ConvexHull(points)

            # Extract the vertices of the convex hull
            hull_vertices = points[hull.vertices]

            # Convert the vertices back to a list of tuples
            points = [tuple(point) for point in hull_vertices]

            #HaoNV35 Start.
            grid_size = calculate_grid_size()
            # grid_points = generate_grid(points, int(distance))
            grid_points = generate_waypoints(points, grid_size[i])
            #HaoNV35 End.

            areas_dot.append(grid_points)

        grid_GPS = []
        for i, area in enumerate(areas_dot):
            area = areas_dot[i]
            unrotated_area = []
            for point in area:
                # convert back in previous coordinate
                new_point = revert_rotate_and_shift_point(
                    point[0],
                    point[1],
                    (-angle),
                    midpoint[0],
                    midpoint[1],
                    (-midpoint[0]),
                    (-midpoint[1]),
                    clockwise=True,
                )
                unrotated_area.append(new_point)
            per_GPS_list = []
            for point in unrotated_area:
                new = calculate_new_lat_lon(min_lat, min_lon, point[1], point[0])
                per_GPS_list.append(new)
            grid_GPS.append(per_GPS_list)

        print("Grid GPS: ", grid_GPS)
        return grid_GPS


def generate_grid(vertices, spacing_m):
    """
    Generate grid points within the polygon defined by `vertices` in a Cartesian coordinate system,
    where the vertices are specified in meters.

    :param vertices: List of (x, y) tuples for the vertices of the polygon.
    :param spacing_m: Distance between points in the grid, in meters.
    :return: List of (x, y) tuples for the grid points inside the polygon.
    """
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)

    points = []
    for i in range(int((max_y - min_y) / spacing_m) + 1):
        for j in range(int((max_x - min_x) / spacing_m) + 1):
            x = min_x + (j * spacing_m)
            y = min_y + (i * spacing_m)
            if ray_casting_point_in_polygon((x, y), vertices):
                points.append((x, y))

    return points
# duc start
# ================= ALGORITHMS ==================
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 * mag2 == 0:
        return 0
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.acos(cos_angle)


def zigzag_path(points, start):
    rows = {}
    for p in points:
        rows.setdefault(round(p[1], 3), []).append(p)
    sorted_rows = sorted(rows.keys())
    path = []
    for i, y in enumerate(sorted_rows):
        row = sorted(rows[y], key=lambda p: p[0])
        if i % 2 != 0:
            row.reverse()
        path += row
    return path

def find_path(points, start, turn_threshold=math.pi/6):
    unvisited = points.copy()
    path = [start]
    curr = start
    last_dir = None
    while unvisited:
        if last_dir is None:
            candidates = unvisited[:]
        else:
            no_turn = []
            for p in unvisited:
                new_dir = (p[0]-curr[0], p[1]-curr[1])
                angle = angle_between(last_dir, new_dir)
                if angle <= turn_threshold:
                    no_turn.append(p)
            candidates = no_turn if no_turn else unvisited
        best_pt = min(candidates, key=lambda p: distance(p, curr))
        best_dir = (best_pt[0]-curr[0], best_pt[1]-curr[1])
        path.append(best_pt)
        unvisited.remove(best_pt)
        curr = best_pt
        last_dir = best_dir
    return path

def nn_2opt_path(points, start):
    unvisited = points.copy()
    path = []
    curr = start
    while unvisited:
        nearest = min(unvisited, key=lambda p: distance(p, curr))
        path.append(nearest)
        unvisited.remove(nearest)
        curr = nearest
    # simple 2-opt
    improved = True
    while improved:
        improved = False
        n = len(path)
        for i in range(n-1):
            for j in range(i+2, n):
                if j == n-1 and i == 0:
                    continue
                a, b = path[i], path[i+1]
                c, d = path[j], path[(j+1) % n] if (j+1)<n else None
                old = distance(a,b) + (distance(c,d) if d else 0)
                new = distance(a,c) + (distance(b,d) if d else 0)
                if new + 1e-6 < old:
                    path[i+1:j+1] = reversed(path[i+1:j+1])
                    improved = True
    return path

def sa_path(points, start, iterations=1000):
    path = points.copy()
    random.shuffle(path)
    def cost(pth):
        dist = 0
        curr = start
        for p in pth:
            dist += distance(p, curr)
            curr = p
        return dist
    T = 100.0
    alpha = 0.995
    for _ in range(iterations):
        i, j = sorted(random.sample(range(len(path)), 2))
        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
        dE = cost(new_path) - cost(path)
        if dE < 0 or math.exp(-dE/T) > random.random():
            path = new_path
        T *= alpha
    return path

def dubins_path(points, start):
    return sorted(points, key=lambda p: math.atan2(p[1]-start[1], p[0]-start[0]))


def aco_path(points, start, ants=20, iterations=50, alpha=1, beta=3, rho=0.1, Q=100):
    """
    ACO (Ant Colony Optimization) cho bài toán tìm đường UAV qua các điểm grid.
    
    Args:
        points: danh sách các điểm cần thăm [(x,y), ...]
        start: điểm xuất phát (x, y)
        ants: số lượng kiến mô phỏng
        iterations: số vòng lặp cập nhật pheromone
        alpha: trọng số pheromone
        beta: trọng số heuristic (1/khoảng cách)
        rho: hệ số bay hơi pheromone (0–1)
        Q: hằng số điều chỉnh lượng pheromone lắng đọng
        
    Returns:
        best_path: danh sách điểm theo thứ tự tối ưu
    """
    n = len(points)
    all_points = [start] + points
    dist = [[distance(a, b) for b in all_points] for a in all_points]

    # Khởi tạo pheromone
    tau = [[1.0 for _ in range(n+1)] for _ in range(n+1)]

    best_path = []
    best_length = float('inf')

    for it in range(iterations):
        all_ant_paths = []
        all_ant_lengths = []

        for _ in range(ants):
            unvisited = set(range(1, n+1))
            curr = 0  # start index (0 = start)
            path = [curr]
            length = 0

            # Mỗi kiến di chuyển cho đến khi thăm hết các điểm
            while unvisited:
                probs = []
                for j in unvisited:
                    tau_ij = tau[curr][j] ** alpha
                    eta_ij = (1 / dist[curr][j]) ** beta if dist[curr][j] > 0 else 0
                    probs.append((j, tau_ij * eta_ij))
                total = sum(p for _, p in probs)
                if total == 0:
                    next_j = random.choice(list(unvisited))
                else:
                    # chọn điểm tiếp theo theo xác suất
                    r = random.random() * total
                    cum = 0
                    for j, p in probs:
                        cum += p
                        if cum >= r:
                            next_j = j
                            break

                path.append(next_j)
                length += dist[curr][next_j]
                curr = next_j
                unvisited.remove(next_j)

            all_ant_paths.append(path)
            all_ant_lengths.append(length)

            # Cập nhật đường đi tốt nhất
            if length < best_length:
                best_length = length
                best_path = path

        # Bay hơi pheromone
        for i in range(n+1):
            for j in range(n+1):
                tau[i][j] *= (1 - rho)

        # Lắng đọng pheromone
        for k, path in enumerate(all_ant_paths):
            Lk = all_ant_lengths[k]
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                tau[a][b] += Q / Lk
                tau[b][a] += Q / Lk  # đối xứng

        # Có thể in tiến trình để quan sát
        print(f"Iteration {it+1}/{iterations}, best length = {best_length:.2f}")

    # Trả về đường đi tối ưu theo thứ tự điểm (bỏ index 0)
    best_path_points = [all_points[i] for i in best_path[1:]]
    return best_path_points

def ga_path(points, start, pop_size=50, generations=300, mutation_rate=0.1, elite_size=5):
    """
    Thuật toán di truyền (Genetic Algorithm) để tối ưu đường đi UAV.
    - points: danh sách tọa độ điểm
    - start: điểm bắt đầu
    - pop_size: kích thước quần thể
    - generations: số thế hệ lặp
    - mutation_rate: xác suất đột biến
    - elite_size: số cá thể tốt nhất giữ lại mỗi thế hệ
    """

    # --- 1. Hàm tính tổng chiều dài quãng đường ---
    def total_distance(route):
        dist = 0
        curr = start
        for p in route:
            dist += distance(curr, p)
            curr = p
        return dist

    # --- 2. Khởi tạo quần thể ban đầu ---
    population = []
    for _ in range(pop_size):
        individual = points[:]  # copy danh sách điểm
        random.shuffle(individual)
        population.append(individual)

    # --- 3. Hàm chọn lọc: chọn cá thể tốt nhất ---
    def selection(pop):
        ranked = sorted(pop, key=lambda r: total_distance(r))
        return ranked[:elite_size]  # giữ lại elite_size cá thể tốt nhất

    # --- 4. Lai ghép (crossover) giữa 2 cá thể ---
    def crossover(p1, p2):
        a, b = sorted(random.sample(range(len(p1)), 2))
        child = [None]*len(p1)
        child[a:b] = p1[a:b]
        fill = [x for x in p2 if x not in child]
        idx = 0
        for i in range(len(p1)):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    # --- 5. Đột biến (mutation) ---
    def mutate(route):
        for i in range(len(route)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(route)-1)
                route[i], route[j] = route[j], route[i]
        return route

    # --- 6. Tiến hóa qua nhiều thế hệ ---
    best_route = None
    best_dist = float('inf')

    for _ in range(generations):
        selected = selection(population)
        new_pop = selected[:]  # giữ lại elite
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

        # cập nhật cá thể tốt nhất
        curr_best = min(population, key=lambda r: total_distance(r))
        curr_dist = total_distance(curr_best)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_route = curr_best[:]

    return best_route


def abc_path(points, start, colony_size=30, limit=20, iterations=100):
    """
    ABC (Artificial Bee Colony) cho bài toán tìm đường UAV.
    Mỗi "bee" đại diện cho một route (hoán vị các điểm).
    """
    n = len(points)

    def total_distance(route):
        dist = 0
        curr = start
        for p in route:
            dist += distance(curr, p)
            curr = p
        return dist

    # --- Khởi tạo quần thể ---
    food_sources = [random.sample(points, n) for _ in range(colony_size)]
    fitness = [1 / (1 + total_distance(p)) for p in food_sources]
    trial = [0] * colony_size

    best_route = min(food_sources, key=lambda r: total_distance(r))
    best_dist = total_distance(best_route)

    for it in range(iterations):
        # --- Pha Employed Bee ---
        for i in range(colony_size):
            k = random.choice([x for x in range(colony_size) if x != i])
            new_solution = food_sources[i][:]
            a, b = random.sample(range(n), 2)
            new_solution[a], new_solution[b] = new_solution[b], new_solution[a]
            if total_distance(new_solution) < total_distance(food_sources[i]):
                food_sources[i] = new_solution
                trial[i] = 0
            else:
                trial[i] += 1

        # --- Pha Onlooker Bee ---
        prob = [f / sum(fitness) for f in fitness]
        for i in range(colony_size):
            if random.random() < prob[i]:
                k = random.choice([x for x in range(colony_size) if x != i])
                new_solution = food_sources[i][:]
                a, b = random.sample(range(n), 2)
                new_solution[a], new_solution[b] = new_solution[b], new_solution[a]
                if total_distance(new_solution) < total_distance(food_sources[i]):
                    food_sources[i] = new_solution
                    trial[i] = 0
                else:
                    trial[i] += 1

        # --- Pha Scout Bee ---
        for i in range(colony_size):
            if trial[i] > limit:
                food_sources[i] = random.sample(points, n)
                trial[i] = 0

        # Cập nhật fitness và cá thể tốt nhất
        fitness = [1 / (1 + total_distance(p)) for p in food_sources]
        curr_best = min(food_sources, key=lambda r: total_distance(r))
        curr_dist = total_distance(curr_best)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_route = curr_best[:]

        print(f"ABC Iter {it+1}/{iterations}: best distance = {best_dist:.3f}")

    return best_route

def ga_path_with_turns(points, start, pop_size=50, generations=200, mutation_rate=0.1, elite_size=5):
    """
    GA tối ưu đường đi: hàm fitness = distance + turn_penalty
    """
    def total_cost(route):
        cost = 0
        curr = start
        prev_vector = None
        for p in route:
            cost += distance(curr, p)
            vector = (p[0]-curr[0], p[1]-curr[1])
            if prev_vector:
                # tính cosine góc giữa 2 vector → tăng penalty nếu quay
                dot = prev_vector[0]*vector[0] + prev_vector[1]*vector[1]
                mag = math.hypot(*prev_vector) * math.hypot(*vector)
                if mag > 0:
                    cos_angle = dot / mag
                    if abs(cos_angle) < 0.99:  # góc lớn → tính như quay
                        cost += 0.1 * distance(curr, p)  # thêm penalty
            prev_vector = vector
            curr = p
        return cost
    
    # --- Khởi tạo quần thể ---
    population = [random.sample(points, len(points)) for _ in range(pop_size)]
    
    best_route = None
    best_cost_val = float('inf')
    
    for _ in range(generations):
        # Chọn lọc
        ranked = sorted(population, key=total_cost)
        new_pop = ranked[:elite_size]  # giữ elite
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(ranked[:elite_size], 2)
            a, b = sorted(random.sample(range(len(p1)), 2))
            # Order crossover
            child = [None]*len(p1)
            child[a:b] = p1[a:b]
            fill = [x for x in p2 if x not in child]
            idx = 0
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = fill[idx]
                    idx += 1
            # Mutation
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        population = new_pop
        # Cập nhật best
        curr_best = min(population, key=total_cost)
        curr_cost = total_cost(curr_best)
        if curr_cost < best_cost_val:
            best_cost_val = curr_cost
            best_route = curr_best[:]
    
    return best_route
def astar_path_with_turns(points, start):
    """
    A* giữa grid points, chi phí = distance + turn penalty
    """
    unvisited = set(points)
    path = [start]
    curr = start
    prev_vector = None

    while unvisited:
        best_p = None
        best_cost = float('inf')
        for p in unvisited:
            vector = (p[0]-curr[0], p[1]-curr[1])
            cost = distance(curr, p)
            if prev_vector:
                dot = prev_vector[0]*vector[0] + prev_vector[1]*vector[1]
                mag = math.hypot(*prev_vector) * math.hypot(*vector)
                if mag > 0:
                    cos_angle = dot / mag
                    if abs(cos_angle) < 0.99:
                        cost += 0.1 * distance(curr, p)
            if cost < best_cost:
                best_cost = cost
                best_p = p
        path.append(best_p)
        unvisited.remove(best_p)
        prev_vector = (best_p[0]-curr[0], best_p[1]-curr[1])
        curr = best_p

    return path

# --- cost function (distance + penalty for turns > 10 deg) ---
def calculate_cost_for_path(path, turn_angle_threshold_deg=10.0, turn_penalty=100.0):
    if not path or len(path) == 1:
        return 0.0, 0.0, 0  # cost, total_dist, turns
    total_dist = 0.0
    turns = 0
    last_dir = None
    for i in range(1, len(path)):
        a = path[i-1]
        b = path[i]
        d = distance(a, b)
        total_dist += d
        new_dir = (b[0]-a[0], b[1]-a[1])
        if last_dir is not None:
            ang = math.degrees(angle_between(last_dir, new_dir))
            if ang > turn_angle_threshold_deg:
                turns += 1
        last_dir = new_dir
    cost = total_dist + turns * turn_penalty
    return cost, total_dist, turns
# --- wrap your provided algorithms (assuming these functions exist in scope)
# If you already defined find_path, nn_2opt_path, sa_path, dubins_path in your module,
# these calls will use them. If not, paste your implementations before this function.

def generate_waypoints(area_vertices, grid_size):
    print("===================================================================================")
    area_min_x = min(v[0] for v in area_vertices)
    area_max_x = max(v[0] for v in area_vertices)
    area_min_y = min(v[1] for v in area_vertices)
    area_max_y = max(v[1] for v in area_vertices)
    print("vertices: ", area_vertices)
    print("min_x, max_x, min_y, max_y: ", area_min_x, area_max_x, area_min_y, area_max_y)
    area_width = area_max_x - area_min_x
    area_height = area_max_y - area_min_y
    print("Area width, height: ", area_width, area_height)

    default_grid_width = grid_size[0]
    default_grid_height = grid_size[1]
    print("Grid width, height: ", default_grid_width, default_grid_height)

    longest_edge_length, longest_edge_coord = find_longest_edge(area_vertices)
    print("Coord, longest_edge_length: ", longest_edge_coord, longest_edge_length)
    if longest_edge_coord[0][0] < longest_edge_coord[1][0]:
        x_root_coord = longest_edge_coord[0][0]
        y_root_coord = longest_edge_coord[0][1]
    else:
        x_root_coord = longest_edge_coord[1][0]
        y_root_coord = longest_edge_coord[1][1]

    number_of_rows = int(area_height/default_grid_height) + 1
    print("Number of rows: ", number_of_rows)
    new_grid_height = area_height / number_of_rows

    intersection_points, segment_length, flag = find_parallel_polygon_intersection(area_vertices, new_grid_height, number_of_rows)

    new_grid_width = []
    root_grid = (longest_edge_length - default_grid_width) / (int(area_width / default_grid_width))
    new_grid_width.insert(0, root_grid)
    for i in range(len(segment_length)):
        row_grid_width = (segment_length[i] - default_grid_width) / (int(segment_length[i] / default_grid_width))
        new_grid_width.append(row_grid_width)
    #Write a new function here

    print("New grid width: ", new_grid_width)
    print("New grid height: ", new_grid_height)

    starting_points = []
    for i in range(1, len(intersection_points), 2):
        p1 = intersection_points[i]
        p2 = intersection_points[i - 1]
        if p1[0] < p2[0]:
            starting_points.append(p1)
        else:
            starting_points.append(p2)
    print("Starting points: ", starting_points)
    #Write a new function here

    points = []
    segment_length.insert(0, longest_edge_length)
    for i in range(number_of_rows): 
        for j in range(int(segment_length[i]/default_grid_width) + 1):
            if flag:
                if 0 == i:
                    x = x_root_coord + default_grid_width / 2 + (j * new_grid_width[i])
                    y = y_root_coord + new_grid_height / 2 
                else:
                    x = starting_points[i-1][0] + default_grid_width/2 + (j * new_grid_width[i])
                    y = starting_points[i-1][1] + new_grid_height/2
            else:
                if 0 == i:
                    x = x_root_coord + default_grid_width / 2 + (j * new_grid_width[i])
                    y = y_root_coord - new_grid_height / 2 
                else:
                    x = starting_points[i-1][0] + default_grid_width/2 + (j * new_grid_width[i])
                    y = starting_points[i-1][1] - new_grid_height/2
            # if ray_casting_point_in_polygon((x, y), vertices):
            points.append((x, y))
    print("Generated points: ", points)

    index = 1
    final_list = []
    for i in range(number_of_rows):
        sub_list = []
        sub_list.append(points[index-1])
        while index != len(points) and points[index][1] == points[index-1][1]:
            sub_list.append(points[index])
            index += 1
        print("Sub list: ", sub_list)
        index += 1
        if i % 2 != 0:
            sub_list.reverse()
        final_list.append(sub_list)
    final_final_list = []
    for i in range(number_of_rows):
        final_final_list = final_final_list + final_list[i]
    print("Final list (Zig-zag): ", final_final_list)

    # ---------- START SIMULATION: compare multiple path algorithms ----------
    # Use the zigzag path produced above as the zigzag algorithm's output
    zigzag_path = final_final_list.copy()

    # Start point for simulation: use first point of the first row (as you requested)
    if len(zigzag_path) == 0:
        return []  # no points
    start_point = zigzag_path[0]

    # Prepare other algorithm outputs (assumes these functions are defined in your code)
    # They should return a list/path of points (in same coordinate system as `points`)
    try:
        path_find, = find_path(points.copy(), start_point) if isinstance(find_path(points.copy(), start_point), tuple) else find_path(points.copy(), start_point)
    except Exception:
        # Some implementations of find_path might return (path,) or path; handle gracefully
        try:
            path_find = find_path(points.copy(), start_point)
        except Exception:
            path_find = []  # fallback

    # For nn_2opt_path, sa_path, dubins_path: they should return lists
    try:
        path_nn2opt = nn_2opt_path(points.copy(), start_point)
    except Exception:
        path_nn2opt = []

    try:
        path_sa = sa_path(points.copy(), start_point)
    except Exception:
        path_sa = []

    try:
        path_dubins = dubins_path(points.copy(), start_point)
    except Exception:
        path_dubins = []
    try:
        path_aco = aco_path(points.copy(), start_point)
    except Exception:
        path_aco = []
    try:
        path_ga = ga_path(points.copy(), start_point)
    except Exception:
        path_ga = []
    try:
        path_abc = abc_path(points.copy(), start_point)
    except Exception:
        path_abc = []
    try:
        path_ga_with_turns = ga_path_with_turns(points.copy(), start_point)
    except Exception:
        path_ga_with_turns = []
    try:
        path_A = astar_path_with_turns(points.copy(), start_point)
    except Exception:
        path_A = []
    # Some algorithms (like your find_path) might include the start as first element; ensure consistency:
    # All algorithm outputs should be lists of points; if any returned path doesn't include start_point as first,
    # we keep them as-is because cost function computes distances sequentially.

    algos = {
        "zigzag": zigzag_path,
        "find_path": path_find,
        "nn_2opt": path_nn2opt,
        "sa": path_sa,
        "dubins": path_dubins,
        "ACO": path_aco,
        "GA": path_ga,
        "ABC": path_abc,
        "GA_with_turn": path_ga_with_turns,
        "A*_cai_tien": path_A

    }

    best_name = None
    best_cost_val = float("inf")
    best_path = None
    for name, path in algos.items():
        if not path:
            continue
        cost, tot_dist, turns = calculate_cost_for_path(path)
        print(f"Algorithm {name}: cost={cost:.3f}, dist={tot_dist:.3f}, turns={turns}")
        if cost < best_cost_val:
            best_cost_val = cost
            best_name = name
            best_path = path

    print(f"==> ✅Selected algorithm: {best_name} (cost={best_cost_val:.3f})")
    # Return the chosen ordered list of grid points (in the same coordinate system as points)
    return best_path

# Duc end

# def generate_waypoints(area_vertices, grid_size):
#     print("===================================================================================")
#     area_min_x = min(v[0] for v in area_vertices)
#     area_max_x = max(v[0] for v in area_vertices)
#     area_min_y = min(v[1] for v in area_vertices)
#     area_max_y = max(v[1] for v in area_vertices)
#     print("vertices: ", area_vertices)
#     print("min_x, max_x, min_y, max_y: ", area_min_x, area_max_x, area_min_y, area_max_y)
#     area_width = area_max_x - area_min_x
#     area_height = area_max_y - area_min_y
#     print("Area width, height: ", area_width, area_height)

#     default_grid_width = grid_size[0]
#     default_grid_height = grid_size[1]
#     print("Grid width, height: ", default_grid_width, default_grid_height)

#     longest_edge_length, longest_edge_coord = find_longest_edge(area_vertices)
#     print("Coord, longest_edge_length: ", longest_edge_coord, longest_edge_length)
#     if longest_edge_coord[0][0] < longest_edge_coord[1][0]:
#         x_root_coord = longest_edge_coord[0][0]
#         y_root_coord = longest_edge_coord[0][1]
#     else:
#         x_root_coord = longest_edge_coord[1][0] 
#         y_root_coord = longest_edge_coord[1][1]

#     number_of_rows = int(area_height/default_grid_height) + 1
#     print("Number of rows: ", number_of_rows)
#     new_grid_height = area_height / number_of_rows

#     intersection_points, segment_length, flag = find_parallel_polygon_intersection(area_vertices, new_grid_height, number_of_rows)

#     new_grid_width = []
#     root_grid = (longest_edge_length - default_grid_width) / (int(area_width / default_grid_width))
#     new_grid_width.insert(0, root_grid)
#     for i in range(len(segment_length)):
#         row_grid_width = (segment_length[i] - default_grid_width) / (int(segment_length[i] / default_grid_width))
#         new_grid_width.append(row_grid_width)
#     #Write a new function here

#     print("New grid width: ", new_grid_width)
#     print("New grid height: ", new_grid_height)

#     starting_points = []
#     for i in range(1, len(intersection_points), 2):
#         p1 = intersection_points[i]
#         p2 = intersection_points[i - 1]
#         if p1[0] < p2[0]:
#             starting_points.append(p1)
#         else:
#             starting_points.append(p2)
#     print("Starting points: ", starting_points)
#     #Write a new function here

#     points = []
#     segment_length.insert(0, longest_edge_length)
#     for i in range(number_of_rows): 
#         for j in range(int(segment_length[i]/default_grid_width) + 1):
#             if flag:
#                 if 0 == i:
#                     x = x_root_coord + default_grid_width / 2 + (j * new_grid_width[i])
#                     y = y_root_coord + new_grid_height / 2 
#                 else:
#                     x = starting_points[i-1][0] + default_grid_width/2 + (j * new_grid_width[i])
#                     y = starting_points[i-1][1] + new_grid_height/2
#             else:
#                 if 0 == i:
#                     x = x_root_coord + default_grid_width / 2 + (j * new_grid_width[i])
#                     y = y_root_coord - new_grid_height / 2 
#                 else:
#                     x = starting_points[i-1][0] + default_grid_width/2 + (j * new_grid_width[i])
#                     y = starting_points[i-1][1] - new_grid_height/2
#             # if ray_casting_point_in_polygon((x, y), vertices):
#             points.append((x, y))
#     print("Generated points: ", points)

#     index = 1
#     final_list = []
#     for i in range(number_of_rows):
#         sub_list = []
#         sub_list.append(points[index-1])
#         while index != len(points) and points[index][1] == points[index-1][1]:
#             sub_list.append(points[index])
#             index += 1
#         print("Sub list: ", sub_list)
#         index += 1
#         if i % 2 != 0:
#             sub_list.reverse()
#         final_list.append(sub_list)
#     final_final_list = []
#     for i in range(number_of_rows):
#         final_final_list = final_final_list + final_list[i]
#     print("Final list: ", final_final_list)
#     return final_final_list

#HaoNV35 Start.
def calculate_grid_size():
    uav_num = 5
    h_fov = (90, 90, 100, 100, 100)
    v_fov = (52, 52, 52, 52, 52)
    uav_alt = (10, 10, 10, 10, 10)
    h_overlap = 0
    v_overlap = 0
    grid_size = []  
    for i in range(uav_num): 
        grid_width, grid_height = calculate_grid_size_from_hfov_and_vfov(h_fov[i], v_fov[i], uav_alt[i])
        overlapped_grid_width, overlapped_grid_height = calculate_overlapped_grid_size(grid_width, grid_height, h_overlap, v_overlap)
        grid_size.append((overlapped_grid_width, overlapped_grid_height))
    # print(grid_size)
    return grid_size
#HaoNV35 End.

def remove_duplicate_pts(vertices):
    """
    Remove duplicate points from a list of vertices.

    :param vertices: List of (x, y) tuples for the vertices.
    :return: List of (x, y) tuples for the vertices with duplicates removed.
    """
    temp = []
    for i in vertices:
        if i not in temp:
            temp.append(i)
    return temp


if __name__ == "__main__":
    pass
