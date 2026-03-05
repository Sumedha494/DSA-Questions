#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def set_zeroes_optimal(matrix):
    """
    Sets matrix zeroes in-place using O(1) extra space.
    """
    if not matrix:
        return

    rows = len(matrix)
    cols = len(matrix[0])

    first_row_zero = False
    first_col_zero = False

    # 1. Check if first column has any zeros
    for r in range(rows):
        if matrix[r][0] == 0:
            first_col_zero = True
            break

    # 2. Check if first row has any zeros
    for c in range(cols):
        if matrix[0][c] == 0:
            first_row_zero = True
            break

    # 3. Use first row and column as markers
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                matrix[r][0] = 0  # Mark row header
                matrix[0][c] = 0  # Mark col header

    # 4. Set elements to zero based on markers
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    # 5. Handle first row and first column
    if first_col_zero:
        for r in range(rows):
            matrix[r][0] = 0

    if first_row_zero:
        for c in range(cols):
            matrix[0][c] = 0

# --- Example Usage ---
matrix = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

print("Original:")
for row in matrix: print(row)

set_zeroes_optimal(matrix)

print("\nResult:")
for row in matrix: print(row)


# In[ ]:


def set_zeroes_simple(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    rows_to_zero = set()
    cols_to_zero = set()

    # Pass 1: Find zeros
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                rows_to_zero.add(r)
                cols_to_zero.add(c)

    # Pass 2: Update matrix
    for r in range(rows):
        for c in range(cols):
            if r in rows_to_zero or c in cols_to_zero:
                matrix[r][c] = 0

# --- Example ---
m = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
set_zeroes_simple(m)
for row in m: print(row)

