#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def rotateMatrix(matrix):
    """
    Rotate matrix 90 degrees clockwise (to right)

    Approach: Transpose + Reverse each row
    Time: O(n²)
    Space: O(1) - in-place
    """
    if not matrix:
        return matrix

    n = len(matrix)

    # Step 1: Transpose (swap rows with columns)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    for i in range(n):
        matrix[i].reverse()

    return matrix


def rotateMatrix_detailed(matrix):
    """
    Step-by-step explanation
    """
    print("Rotate Matrix 90° Clockwise")
    print("=" * 50)
    print("Original Matrix:")
    printMatrix(matrix)

    n = len(matrix)

    # Step 1: Transpose
    print("\nStep 1: Transpose (swap rows & columns)")
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    printMatrix(matrix)

    # Step 2: Reverse rows
    print("\nStep 2: Reverse each row")
    for i in range(n):
        matrix[i].reverse()

    printMatrix(matrix)

    print("\n" + "=" * 50)
    print("Rotated Matrix (90° Right):")
    printMatrix(matrix)

    return matrix


def printMatrix(matrix):
    """
    Print matrix nicely
    """
    for row in matrix:
        print(" ", row)


def rotateMatrix_new_matrix(matrix):
    """
    Using new matrix (easier to understand)
    Time: O(n²), Space: O(n²)
    """
    if not matrix:
        return matrix

    n = len(matrix)
    result = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # New position: row -> col, col -> (n-1-row)
            result[j][n - 1 - i] = matrix[i][j]

    return result


def rotateMatrix_anticlockwise(matrix):
    """
    Rotate 90° anti-clockwise (to left)
    Approach: Transpose + Reverse each column
    """
    if not matrix:
        return matrix

    n = len(matrix)

    # Step 1: Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each column (or reverse row order)
    matrix.reverse()

    return matrix


def rotateMatrix_180(matrix):
    """
    Rotate 180 degrees
    """
    if not matrix:
        return matrix

    n = len(matrix)

    # Reverse rows, then reverse each row
    matrix.reverse()
    for row in matrix:
        row.reverse()

    return matrix


def rotateMatrix_k_times(matrix, k):
    """
    Rotate k times (90° each)
    """
    k = k % 4  # Only 4 unique rotations

    for _ in range(k):
        rotateMatrix(matrix)

    return matrix


def visualize_rotation(matrix):
    """
    Visual rotation process
    """
    import copy

    print("Rotation Visualization")
    print("=" * 60)

    original = copy.deepcopy(matrix)

    print("Original:")
    printMatrix(original)

    print("\nRotation Logic:")
    print("  (0,0) -> (0,n-1)")
    print("  (0,1) -> (1,n-1)")
    print("  (i,j) -> (j, n-1-i)")

    n = len(matrix)

    # Show where each element goes
    print("\nElement Movement:")
    for i in range(n):
        for j in range(n):
            new_i = j
            new_j = n - 1 - i
            val = original[i][j]
            print("  [" + str(i) + "][" + str(j) + "]=" + str(val), 
                  "->", 
                  "[" + str(new_i) + "][" + str(new_j) + "]")

    # Rotate
    rotateMatrix(matrix)

    print("\nAfter Rotation:")
    printMatrix(matrix)


def rotate_rectangular(matrix):
    """
    Rotate rectangular matrix (m x n)
    Returns new matrix (n x m)
    """
    if not matrix:
        return matrix

    m = len(matrix)
    n = len(matrix[0])

    # New matrix is n x m
    result = [[0] * m for _ in range(n)]

    for i in range(m):
        for j in range(n):
            result[j][m - 1 - i] = matrix[i][j]

    return result


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 50)
    print("TEST 1: Basic 3x3 Matrix")
    print("=" * 50)

    matrix1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Original:")
    printMatrix(matrix1)

    rotateMatrix(matrix1)

    print("\nRotated 90° Right:")
    printMatrix(matrix1)

    print("\n" + "=" * 50)
    print("TEST 2: Detailed Step-by-Step")
    print("=" * 50)

    matrix2 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    rotateMatrix_detailed(matrix2)

    print("\n" + "=" * 50)
    print("TEST 3: 4x4 Matrix")
    print("=" * 50)

    matrix3 = [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16]
    ]

    print("Original:")
    printMatrix(matrix3)

    rotateMatrix(matrix3)

    print("\nRotated:")
    printMatrix(matrix3)

    print("\n" + "=" * 50)
    print("TEST 4: Anti-clockwise Rotation")
    print("=" * 50)

    matrix4 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Original:")
    printMatrix(matrix4)

    rotateMatrix_anticlockwise(matrix4)

    print("\nRotated 90° Left:")
    printMatrix(matrix4)

    print("\n" + "=" * 50)
    print("TEST 5: 180° Rotation")
    print("=" * 50)

    matrix5 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Original:")
    printMatrix(matrix5)

    rotateMatrix_180(matrix5)

    print("\nRotated 180°:")
    printMatrix(matrix5)

    print("\n" + "=" * 50)
    print("TEST 6: Multiple Rotations")
    print("=" * 50)

    matrix6 = [
        [1, 2],
        [3, 4]
    ]

    print("Original:")
    printMatrix(matrix6)

    for k in range(1, 5):
        matrix_copy = [[1, 2], [3, 4]]
        rotateMatrix_k_times(matrix_copy, k)
        print("\nAfter", k, "rotation(s):")
        printMatrix(matrix_copy)

    print("\n" + "=" * 50)
    print("TEST 7: Visualization")
    print("=" * 50)

    matrix7 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    visualize_rotation(matrix7)

    print("\n" + "=" * 50)
    print("TEST 8: Rectangular Matrix")
    print("=" * 50)

    matrix8 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]

    print("Original (2x4):")
    printMatrix(matrix8)

    result8 = rotate_rectangular(matrix8)

    print("\nRotated (4x2):")
    printMatrix(result8)

    print("\n" + "=" * 50)
    print("TEST 9: Edge Cases")
    print("=" * 50)

    # 1x1 matrix
    matrix9 = [[5]]
    print("1x1 Matrix:", matrix9)
    rotateMatrix(matrix9)
    print("Rotated:   ", matrix9)

    # 2x2 matrix
    matrix10 = [[1, 2], [3, 4]]
    print("\n2x2 Matrix:")
    printMatrix(matrix10)
    rotateMatrix(matrix10)
    print("Rotated:")
    printMatrix(matrix10)

    print("\n" + "=" * 50)
    print("ALGORITHM SUMMARY")
    print("=" * 50)
    print("""
Rotate Matrix 90 Degrees Clockwise (Right)

Method 1: Transpose + Reverse Rows (In-place)
  Step 1: Transpose matrix
  Step 2: Reverse each row
  Time: O(n^2), Space: O(1)

Method 2: New Matrix
  result[j][n-1-i] = matrix[i][j]
  Time: O(n^2), Space: O(n^2)

Rotations Summary:
  90 Right:  Transpose + Reverse rows
  90 Left:   Transpose + Reverse columns
  180:       Reverse rows + Reverse each row
  270 Right: Same as 90 Left

LeetCode #48: Rotate Image
    """)

    print("\n" + "=" * 50)
    print("VISUAL TRANSFORMATION")
    print("=" * 50)
    print("""
Original:        90° Right:
1  2  3          7  4  1
4  5  6    ->    8  5  2
7  8  9          9  6  3

Step 1 - Transpose:
1  2  3          1  4  7
4  5  6    ->    2  5  8
7  8  9          3  6  9

Step 2 - Reverse Rows:
1  4  7          7  4  1
2  5  8    ->    8  5  2
3  6  9          9  6  3
    """)

