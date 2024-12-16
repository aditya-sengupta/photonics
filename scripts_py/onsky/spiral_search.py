# %%
def slidett(pos): # to be replaced with the real one
    print(pos)

def next_direction(current_direction):
    if current_direction == "e":
        return "n"
    elif current_direction == "n":
        return "w"
    elif current_direction == "w":
        return "s"
    elif current_direction == "s":
        return "e"

def next_cell(current_direction, ns_cell_idx, ew_cell_idx):
    if current_direction == "e":
        ew_cell_idx += 1
    elif current_direction == "w":
        ew_cell_idx -= 1
    elif current_direction == "n":
        ns_cell_idx += 1
    elif current_direction == "s":
        ns_cell_idx -= 1
    
    return ns_cell_idx, ew_cell_idx

def spiral_search(grid_size, grid_step, f_per_cell):
    """
    Executes a spiral search over a square grid of size "grid_size". 
    "grid_step" specifies the spacing between cells in each direction.
    "f_per_cell" is a user-specified function with no arguments to run at each position.
    """
    num_cells_visited = 0
    current_diameter, current_radius = 1, 1
    ns_cell_idx, ew_cell_idx = 0, 0
    current_direction = "e"
    ns, ew = [], []
    while num_cells_visited < grid_size ** 2:
        # visit the current cell
        slidett([ns_cell_idx * grid_step, ew_cell_idx * grid_step])
        f_per_cell()
        num_cells_visited += 1
        ns_to_check, ew_to_check = next_cell(current_direction, ns_cell_idx, ew_cell_idx)
        if abs(ns_to_check) >= current_radius or abs(ew_to_check) >= current_radius: # proposed cell is out of current bounds
            if num_cells_visited < current_diameter ** 2: # there's still more cells within the current radius
                # print(f"Changing direction from {current_direction}")
                current_direction = next_direction(current_direction)
                ns_cell_idx, ew_cell_idx = next_cell(current_direction, ns_cell_idx, ew_cell_idx)
            else: # we've run out of cells in the current radius
                current_diameter += 2
                current_radius += 1
                ns_cell_idx, ew_cell_idx = ns_to_check, ew_to_check
        else:
            ns_cell_idx, ew_cell_idx = ns_to_check, ew_to_check 
            
spiral_search(5, 1, lambda: None)

# %%
