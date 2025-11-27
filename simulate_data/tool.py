import numpy as np


def dda_algorithm(start_p, stop_p, x_vec, y_vec, inv_y_vec, matrix_grid):
    """

    :param start_p: the index of initial point, tuple(int, int)
    :param stop_p: the index of end point, tuple(int, int)

    :return: the list of computed points between initial and final points (including initial and final points)
    """

    x_0, y_0 = coordinate_discretizer(x_vec, y_vec, coordinate=start_p)
    x_n, y_n = coordinate_discretizer(x_vec, y_vec, coordinate=stop_p)

    dx = x_n - x_0
    dy = y_n - y_0

    steps = max(abs(dx), abs(dy))

    if float(steps) == 0.0:
        raise ValueError('start and stop points must be different coordinates!')

    x_increment = dx / float(steps)
    y_increment = dy / float(steps)

    x_new = x_0
    y_new = y_0

    i_new, j_new = converter_to_index(x_vec, y_vec, inv_y_vec, dis_coord=(round(x_new), round(y_new)))

    matrix_grid[i_new, j_new] = 1

    for i in range(steps):
        x_new = x_new + x_increment
        y_new = y_new + y_increment

        i_new, j_new = converter_to_index(x_vec, y_vec, inv_y_vec, dis_coord=(round(x_new), round(y_new)))
        matrix_grid[i_new, j_new] = 1

    return matrix_grid

def converter_to_index(x_vec, y_vec, inv_y_vec, dis_coord):
    x = x_vec[dis_coord[0]]
    # x = min(x_vec, key=lambda a: abs(a - x))
    y = y_vec[dis_coord[1]]
    # y = min(y_vec, key=lambda a: abs(a - y))

    i_index = inv_y_vec.index(min(inv_y_vec, key=lambda a: abs(a - y)))
    j_index = x_vec.index(min(x_vec, key=lambda a: abs(a - x)))

    return i_index, j_index

def coordinate_discretizer(x_vec, y_vec, coordinate):

    x, y = coordinate
    x_step = x_vec.index(min(x_vec, key=lambda a: abs(a - x)))
    y_step = y_vec.index(min(y_vec, key=lambda a: abs(a - y)))

    return x_step, y_step


def get_fouling_grid(fouling_start_point, num_fouling, fouling_width,
                     fouling_height, grid_size, pipe_rad, transducer_crd):

    g_matrix = np.zeros(grid_size)

    if num_fouling > 0:
        x_start, y_start = fouling_start_point

        y_end = y_start + fouling_height

        x_end = x_start + (fouling_width * num_fouling)

        pipe_circumference = 2 * np.pi * pipe_rad
        # x axis grid
        x_grid = list(np.linspace(0, transducer_crd[0], grid_size[0]))
        # y axis grid
        y_grid = list(np.linspace(0, pipe_circumference, grid_size[1]))
        # y axis grid in reverse order, later will be used to handle converting from coordinate to index
        inv_y_grid = list(np.linspace(pipe_circumference, 0, grid_size[1]))

        # difference between each grid point on y axis
        pixel_height = pipe_circumference / float(grid_size[1])

        y_list = list(np.arange(y_start, y_end, pixel_height))
        # to include y_end in the list
        y_list.append(y_end)

        for y_ in y_list:
            g_matrix = np.logical_or(g_matrix, dda_algorithm(start_p=(x_start, y_), stop_p=(x_end, y_),
                                     x_vec=x_grid, y_vec=y_grid, inv_y_vec=inv_y_grid, matrix_grid=g_matrix)
                                     )

    return g_matrix.reshape(g_matrix.astype(int).size)

def get_path_grid(laser_crd, transducer_crd, pipe_rad, N_path, grid_size):
    
    if N_path % 2 == 0:
        is_N_path_even = True
        N_path = N_path+1
    else:
        is_N_path_even = False
        
    N_helical = int(np.ceil(N_path / 2))
    
    x_l, y_l = laser_crd
    x_t, y_t = transducer_crd
    # 2*pi*r = pi * d
    pipe_circumference = 2 * np.pi * pipe_rad

    # x axis grid
    x_grid = list(np.linspace(0, x_t, grid_size[0]))
    # y axis grid
    y_grid = list(np.linspace(0, pipe_circumference, grid_size[1]))
    # y axis grid in reverse order, later will be used to handle converting from coordinate to index
    inv_y_grid = list(np.linspace(pipe_circumference, 0, grid_size[1]))

    # initializing matrix b, the number of rows later will be changed from N_helical to (2*N_helical)+1
    matrix_b = np.zeros((2 * N_helical, grid_size[0] * grid_size[1]))

    path_crd = []
    for i in range(0, N_helical):
        # this function gets the coordinates of start and points of each line for each path
        path_coord = path_analysis(x_l, y_l, x_t, y_t, path_order=i, pipe_cfc=pipe_circumference)
        # temporary matrix to keep track grid values within a path for multiple line
        value_mat = np.zeros((grid_size[1], grid_size[0]))
        for line_coord in path_coord:
            # reading start and end point coordinate
            start_point = line_coord[0]
            end_point = line_coord[1]
            # logical_or to update my temporary matrix with new values of new lines within the same path
            value_mat = np.logical_or(value_mat, dda_algorithm(start_p=start_point, stop_p=end_point,
                                                                x_vec=x_grid, y_vec=y_grid, inv_y_vec=inv_y_grid,
                                                                matrix_grid=value_mat)
                                      )
        # vectorize the grid and put it in the corresponding row to path index
        matrix_b[i, :] = value_mat.reshape(value_mat.astype(int).size)

    for i in range(0, N_helical):
        # this function gets the coordinates of start and points of each line for each path
        path_coord = path_analysis_down(x_l, y_l, x_t, y_t, path_order=i, pipe_cfc=pipe_circumference)
        # temporary matrix to keep track grid values within a path for multiple line
        value_mat = np.zeros((grid_size[1], grid_size[0]))
        for line_coord in path_coord:
            # reading start and end point coordinate
            start_point = line_coord[0]
            end_point = line_coord[1]
            # logical_or to update my temporary matrix with new values of new lines within the same path
            value_mat = np.logical_or(value_mat, dda_algorithm(start_p=start_point, stop_p=end_point,
                                                                x_vec=x_grid, y_vec=y_grid, inv_y_vec=inv_y_grid,
                                                                matrix_grid=value_mat))
        # vectorize the grid and put it in the corresponding row to path index
        matrix_b[(i + N_helical), :] = value_mat.reshape(value_mat.astype(int).size)

    matrix_b = np.delete(matrix_b, N_helical, axis=0)
    
    if is_N_path_even:
        matrix_b = matrix_b[np.argsort(matrix_b.sum(axis=1))][:N_path-1] 
    else:
        matrix_b = matrix_b[np.argsort(matrix_b.sum(axis=1))][:N_path]

    
    return matrix_b, N_helical


def path_analysis(x_l, y_l, x_t, y_t, path_order, pipe_cfc, path_down=False):
    # x and y of start location for each line
    x_start = x_l
    y_start = y_l
    x_start_down = x_l
    y_start_down = y_l

    y_end = pipe_cfc
    y_end_down = 0
    path_crd = []
    if path_order == 0:
        # if direct path
        path_crd.append([(x_l, y_l), (x_t, y_t)])
    else:
        nb_full_line = path_order - 1
        for line in range(0, path_order + 1):
            # if last line of the path
            if line == path_order:
                path_crd.append([(x_start, y_start), (x_t, y_t)])
                if path_down:
                    path_crd.append([(x_start_down, y_start_down), (x_t, y_t)])
            else:
                # projected points
                x_p = x_t
                y_p = pipe_cfc * (nb_full_line + 1) + y_t
                y_p_down = -pipe_cfc * (nb_full_line + 1) + y_t

                x_end = ((x_p - x_start) * pipe_cfc + (y_p * x_start - x_p * y_start)) / (y_p - y_start)
                x_end_down = (-y_p_down * x_start_down + x_p * y_start_down) / (y_start_down - y_p_down)

                path_crd.append([(x_start, y_start), (x_end, y_end)])
                if path_down:
                    path_crd.append([(x_start_down, y_start_down), (x_end_down, y_end_down)])
                x_start = x_end
                x_start_down = x_end_down
                y_start = 0
                y_start_down = pipe_cfc
            if nb_full_line > 0:
                nb_full_line -= 1
    return path_crd


def path_analysis_down(x_l, y_l, x_t, y_t, path_order, pipe_cfc, path_down=True):
    # x and y of start location for each line
    x_start = x_l
    y_start = y_l
    x_start_down = x_l
    y_start_down = y_l

    y_end = pipe_cfc
    y_end_down = 0
    path_crd = []
    #    if path_order == 0:
    #        # if direct path
    #        path_crd.append([(x_l, y_l), (x_t, y_t)])
    #    else:
    nb_full_line = path_order - 1
    for line in range(0, path_order + 1):
        # if last line of the path
        if line == path_order:
            path_crd.append([(x_start_down, y_start_down), (x_t, y_t)])
        else:
            # projected points
            x_p = x_t
            y_p = pipe_cfc * (nb_full_line + 1) + y_t
            y_p_down = -pipe_cfc * (nb_full_line + 1) + y_t

            x_end = ((x_p - x_start) * pipe_cfc + (y_p * x_start - x_p * y_start)) / (y_p - y_start)
            x_end_down = (-y_p_down * x_start_down + x_p * y_start_down) / (y_start_down - y_p_down)

            path_crd.append([(x_start_down, y_start_down), (x_end_down, y_end_down)])

            x_start = x_end
            x_start_down = x_end_down
            y_start = 0
            y_start_down = pipe_cfc
        if nb_full_line > 0:
            nb_full_line -= 1
    return path_crd
