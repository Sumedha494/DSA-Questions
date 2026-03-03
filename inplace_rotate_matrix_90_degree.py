#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def rotate_90_clockwise(matrix):
    """
    Rotates an N x N matrix 90 degrees clockwise in-place.
    Strategy: Transpose -> Reverse Rows
    Time: O(N^2)
    Space: O(1)
    """
    n = len(matrix)

    # Step 1: Transpose (Swap matrix[i][j] with matrix[j][i])
    for i in range(n):
        for j in range(i + 1, n): # Start from i+1 to avoid swapping back
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    for i in range(n):
        matrix[i].reverse()
        # Or manually: matrix[i] = matrix[i][::-1]

def print_matrix(matrix):
    for row in matrix:
        print(row)

# --- Example Usage ---
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print("Original:")
print_matrix(matrix)

rotate_90_clockwise(matrix)

print("\nRotated 90° Clockwise:")
print_matrix(matrix)


# In[ ]:


def rotate_layer_by_layer(matrix):
    """
    Rotates matrix using 4-way swap logic.
    """
    n = len(matrix)

    # Loop through layers (0 to n//2)
    for i in range(n // 2):
        # Loop through elements in the current layer
        for j in range(i, n - 1 - i):

            # Save top
            temp = matrix[i][j]

            # Move left to top
            matrix[i][j] = matrix[n - 1 - j][i]

            # Move bottom to left
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]

            # Move right to bottom
            matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]

            # Move top (saved in temp) to right
            matrix[j][n - 1 - i] = temp

# --- Example Usage ---
matrix_4x4 = [
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16]
]

print("Original 4x4:")
print_matrix(matrix_4x4)

rotate_layer_by_layer(matrix_4x4)

print("\nRotated 90° Clockwise:")
print_matrix(matrix_4x4)


# In[ ]:


def rotate_90_anticlockwise(matrix):
    """
    Rotates 90 degrees anti-clockwise in-place.
    Strategy: Transpose -> Reverse Columns
    """
    n = len(matrix)

    # Step 1: Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse Columns (Swap matrix[i][j] with matrix[n-1-i][j])
    for j in range(n):
        for i in range(n // 2):
            matrix[i][j], matrix[n - 1 - i][j] = matrix[n - 1 - i][j], matrix[i][j]

# --- Example Usage ---
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

rotate_90_anticlockwise(matrix)
print("Rotated 90° Anti-Clockwise:")
print_matrix(matrix)


# In[ ]:


def visualize_rotation(matrix):
    n = len(matrix)
    print("1. Initial State:")
    print_matrix(matrix)

    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    print("\n2. After Transpose (Rows become Cols):")
    print_matrix(matrix)

    # Reverse Rows
    for i in range(n):
        matrix[i].reverse()

    print("\n3. After Reversing Rows (Final Result):")
    print_matrix(matrix)

# Run Visualization
m = [[1, 2], [3, 4]]
visualize_rotation(m)

