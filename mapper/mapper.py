def print_mapping(mapping, timestep):
    print(f"Timestep {timestep}:")
    for row in mapping:
        print("  " + " | ".join(f"{cell:12}" for cell in row))
    print()

def cgra_mult_add_heuristic(nrow, ncol, multiply_terms, res_term):
    mapping_per_timestep = []
    pe_state = [['NOP' for _ in range(ncol)] for _ in range(nrow)]
    value_locations = {}  # name -> (row, col)
    timestep = 0

    # 1. Horizontal Distribution phase
    rounds = [multiply_terms[i:i+nrow] for i in range(0, len(multiply_terms), nrow)]
    col = 0
    loaded_terms = []

    
    for num_round, round_terms in enumerate(rounds):
        # Move existing elements east FIRST (if not first iteration)
        if num_round > 0:
            for name in loaded_terms:
                r, c = value_locations[name]
                pe_state[r][c] = f'MOVE EAST'
                value_locations[name] = (r, c+1)
            mapping_per_timestep.append([row[:] for row in pe_state])
            print_mapping(mapping_per_timestep[-1], timestep)
            pe_state = [['NOP' for _ in range(ncol)] for _ in range(nrow)]
            timestep += 1
            col += 1
        
        # Load new round_terms into column 0
        for i, name in enumerate(round_terms):
            pe_state[i][0] = f'LOAD {name}'
            value_locations[name] = (i, 0)
            loaded_terms.append(name)
        mapping_per_timestep.append([row[:] for row in pe_state])
        print_mapping(mapping_per_timestep[-1], timestep)
        pe_state = [['NOP' for _ in range(ncol)] for _ in range(nrow)]
        timestep += 1

    # 2. Parallel vertical reduction Phase
    col_products = []
    
    # Initialize terms for each column
    terms_per_col = []
    for c in range(col+1):
        terms_in_col = [name for name, (r, cc) in value_locations.items() if cc == c]
        terms_per_col.append(terms_in_col)
    
    # Continue until all columns have only one term each
    while any(len(terms) > 1 for terms in terms_per_col):
        operation_happened = False
        
        # Process all columns in parallel
        for c in range(col+1):
            if len(terms_per_col[c]) > 1:
                # Check if multiplication can happen at row 0
                elements_at_row0 = [name for name in terms_per_col[c] if value_locations[name][0] == 0]
                
                if len(elements_at_row0) >= 2:
                    # Multiply at row 0
                    elem1, elem2 = elements_at_row0[0], elements_at_row0[1]
                    pe_state[0][c] = f'MULT {elem1},{elem2}'
                    new_name = f'({elem1}*{elem2})'
                    value_locations[new_name] = (0, c)
                    # Remove old names from tracking
                    del value_locations[elem1]
                    del value_locations[elem2]
                    # Update terms_per_col
                    terms_per_col[c] = [name for name in terms_per_col[c] if name not in [elem1, elem2]]
                    terms_per_col[c].append(new_name)
                    operation_happened = True
                
                # Move all non-zero-row elements north (simultaneously with multiply if applicable)
                for term_name in terms_per_col[c]:
                    current_row = value_locations[term_name][0]
                    if current_row > 0:  # Can move north
                        pe_state[current_row][c] = 'MOVE NORTH'
                        value_locations[term_name] = (current_row-1, c)
                        operation_happened = True
        
        if operation_happened:
            mapping_per_timestep.append([row[:] for row in pe_state])
            print_mapping(mapping_per_timestep[-1], timestep)
            pe_state = [['NOP' for _ in range(ncol)] for _ in range(nrow)]
            timestep += 1
    
    # Collect final products from each column
    for c in range(col+1):
        col_products.append(terms_per_col[c][0])

    # 4. Parallel horizontal reduction Phase (with overlapped res loading)
    # Track products by their current column positions
    products_at_col = {}
    for c in range(col+1):
        products_at_col[c] = col_products[c]
    
    res_loaded = False
    res_at_10 = False
    
    # Continue until only one product remains at column 0
    while len([p for p in products_at_col.values() if p is not None]) > 1:
        operation_happened = False
        new_products_at_col = {}
        accumulator_at_00 = products_at_col.get(0)
        arriving_product = None
        
        # Load res into (1,0) during horizontal reduction if not already loaded
        if not res_loaded and pe_state[1][0] == 'NOP':
            pe_state[1][0] = f'LOAD {res_term}'
            value_locations[res_term] = (1, 0)
            res_loaded = True
            res_at_10 = True
            operation_happened = True
        
        # Move all products west simultaneously
        for c in range(col+1):
            if products_at_col[c] is not None:
                if c == 0:
                    # Product already at (0,0) - stays as accumulator
                    new_products_at_col[0] = products_at_col[c]
                else:
                    # Move product west
                    pe_state[0][c] = 'MOVE WEST'
                    value_locations[products_at_col[c]] = (0, c-1)
                    new_products_at_col[c-1] = products_at_col[c]
                    new_products_at_col[c] = None  # Clear source position
                    operation_happened = True
                    
                    # Check if this product is arriving at (0,0)
                    if c == 1:  # Moving from column 1 to column 0
                        arriving_product = products_at_col[c]
        
        # If a product arrived at (0,0), multiply it with the accumulator
        if arriving_product is not None and accumulator_at_00 is not None:
            pe_state[0][0] = f'MULT {accumulator_at_00},{arriving_product}'
            new_name = f'({accumulator_at_00}*{arriving_product})'
            value_locations[new_name] = (0, 0)
            # Remove old products from tracking
            del value_locations[accumulator_at_00]
            del value_locations[arriving_product]
            new_products_at_col[0] = new_name
            operation_happened = True
        
        products_at_col = new_products_at_col
        
        if operation_happened:
            mapping_per_timestep.append([row[:] for row in pe_state])
            print_mapping(mapping_per_timestep[-1], timestep)
            pe_state = [['NOP' for _ in range(ncol)] for _ in range(nrow)]
            timestep += 1
    
    # Final product should be at column 0
    product = products_at_col[0]
    
    # 5. Move res from (1,0) to (0,0) and add (optimized accumulation)
    if res_at_10:
        pe_state[1][0] = 'MOVE NORTH'
        value_locations[res_term] = (0, 0)
        mapping_per_timestep.append([row[:] for row in pe_state])
        print_mapping(mapping_per_timestep[-1], timestep)
        pe_state = [['NOP' for _ in range(ncol)] for _ in range(nrow)]
        timestep += 1

    # 6. Add res and product at (0,0)
    pe_state[0][0] = f'ADD {res_term},{product}'
    mapping_per_timestep.append([row[:] for row in pe_state])
    print_mapping(mapping_per_timestep[-1], timestep)
    pe_state = [['NOP' for _ in range(ncol)] for _ in range(nrow)]
    timestep += 1

    # 7. Store result from (0,0)
    pe_state[0][0] = 'STORE RESULT'
    mapping_per_timestep.append([row[:] for row in pe_state])
    print_mapping(mapping_per_timestep[-1], timestep)

    return mapping_per_timestep


# Example usage:
multiply_terms = ['A[b][i][j]', 'B[b][j][k]','C[b][i][j]','D[b][j][k]','E[b][i][j]', 'F[b][j][k]','G[b][i][j]','H[b][j][k]','I_[b][j][k]']
res_term = 'res[i][j]'
cgra_mult_add_heuristic(nrow=3, ncol=3, multiply_terms=multiply_terms, res_term=res_term)