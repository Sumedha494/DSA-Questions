#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def nthSpiralElement(matrix, n):
    """
    Find Nth element in spiral order traversal

    Time: O(rows * cols), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return -1

    rows = len(matrix)
    cols = len(matrix[0])

    if n > rows * cols:
        return -1

    top = 0
    bottom = rows - 1
    left = 0
    right = cols - 1
    count = 0

    while top <= bottom and left <= right:
        # Left to Right
        for i in range(left, right + 1):
            count += 1
            if count == n:
                return matrix[top][i]
        top += 1

        # Top to Bottom
        for i in range(top, bottom + 1):
            count += 1
            if count == n:
                return matrix[i][right]
        right -= 1

        # Right to Left
        if top <= bottom:
            for i in range(right, left - 1, -1):
                count += 1
                if count == n:
                    return matrix[bottom][i]
            bottom -= 1

        # Bottom to Top
        if left <= right:
            for i in range(bottom, top - 1, -1):
                count += 1
                if count == n:
                    return matrix[i][left]
            left += 1

    return -1


def spiralOrder(matrix):
    """
    Get full spiral order traversal
    """
    if not matrix or not matrix[0]:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    result = []
    top = 0
    bottom = rows - 1
    left = 0
    right = cols - 1

    while top <= bottom and left <= right:
        # Left to Right
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1

        # Top to Bottom
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        # Right to Left
        if top <= bottom:
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1

        # Bottom to Top
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result


def nthSpiralElement_detailed(matrix, n):
    """
    Step-by-step explanation
    """
    print("Nth Element of Spiral Matrix")
    print("=" * 60)
    print("Matrix:")
    for row in matrix:
        print(" ", row)
    print("N:", n)
    print()

    if not matrix or not matrix[0]:
        return -1

    rows = len(matrix)
    cols = len(matrix[0])

    if n > rows * cols:
        print("N is too large!")
        return -1

    # Get spiral order
    spiral = spiralOrder(matrix)

    print("Spiral Order:")
    print(spiral)
    print()

    # Show positions
    print("Position by position:")
    for i, val in enumerate(spiral):
        marker = " <-- Nth element" if i + 1 == n else ""
        print("  Position", i + 1, ":", val, marker)
        if i + 1 == n:
            break

    print()
    print("=" * 60)
    print("Nth element (N=" + str(n) + "):", spiral[n - 1])

    return spiral[n - 1]


def visualize_spiral(matrix):
    """
    Visual spiral traversal
    """
    print("Spiral Traversal Visualization")
    print("=" * 60)
    print("Matrix:")
    for row in matrix:
        print(" ", row)
    print()

    rows = len(matrix)
    cols = len(matrix[0])

    spiral = spiralOrder(matrix)

    print("Spiral Order with Directions:")
    print("-" * 40)

    top = 0
    bottom = rows - 1
    left = 0
    right = cols - 1
    idx = 0
    layer = 1

    while top <= bottom and left <= right:
        print("\nLayer", str(layer) + ":")

        # Left to Right
        print("  → Right:", end=" ")
        for i in range(left, right + 1):
            print(matrix[top][i], end=" ")
            idx += 1
        print()
        top += 1

        # Top to Bottom
        if top <= bottom:
            print("  ↓ Down: ", end=" ")
            for i in range(top, bottom + 1):
                print(matrix[i][right], end=" ")
                idx += 1
            print()
        right -= 1

        # Right to Left
        if top <= bottom:
            print("  ← Left: ", end=" ")
            for i in range(right, left - 1, -1):
                print(matrix[bottom][i], end=" ")
                idx += 1
            print()
            bottom -= 1

        # Bottom to Top
        if left <= right:
            print("  ↑ Up:   ", end=" ")
            for i in range(bottom, top - 1, -1):
                print(matrix[i][left], end=" ")
                idx += 1
            print()
        left += 1

        layer += 1

    print()
    print("Complete spiral:", spiral)


def generateSpiralMatrix(n):
    """
    Generate n x n spiral matrix
    """
    matrix = [[0] * n for _ in range(n)]

    top = 0
    bottom = n - 1
    left = 0
    right = n - 1
    num = 1

    while top <= bottom and left <= right:
        for i in range(left, right + 1):
            matrix[top][i] = num
            num += 1
        top += 1

        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1

        if top <= bottom:
            for i in range(right, left - 1, -1):
                matrix[bottom][i] = num
                num += 1
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = num
                num += 1
            left += 1

    return matrix


def findPositionInSpiral(matrix, target):
    """
    Find position of element in spiral order
    """
    spiral = spiralOrder(matrix)

    for i, val in enumerate(spiral):
        if val == target:
            return i + 1

    return -1


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Example")
    print("=" * 60)

    matrix1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Matrix:")
    for row in matrix1:
        print(" ", row)
    print()

    print("Spiral order:", spiralOrder(matrix1))
    print()

    for n in [1, 5, 9]:
        result = nthSpiralElement(matrix1, n)
        print("N=" + str(n) + ":", result)

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    matrix2 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]

    nthSpiralElement_detailed(matrix2, 7)

    print("\n" + "=" * 60)
    print("TEST 3: Visualization")
    print("=" * 60)

    matrix3 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    visualize_spiral(matrix3)

    print("\n" + "=" * 60)
    print("TEST 4: Different Sizes")
    print("=" * 60)

    # 2x2
    matrix4a = [[1, 2], [3, 4]]
    print("2x2 Matrix:", matrix4a)
    print("Spiral:", spiralOrder(matrix4a))
    print()

    # 4x4
    matrix4b = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    print("4x4 Matrix:")
    for row in matrix4b:
        print(" ", row)
    print("Spiral:", spiralOrder(matrix4b))

    print("\n" + "=" * 60)
    print("TEST 5: Single Row/Column")
    print("=" * 60)

    # Single row
    matrix5a = [[1, 2, 3, 4]]
    print("Single row:", matrix5a)
    print("Spiral:", spiralOrder(matrix5a))
    print()

    # Single column
    matrix5b = [[1], [2], [3], [4]]
    print("Single column:", matrix5b)
    print("Spiral:", spiralOrder(matrix5b))

    print("\n" + "=" * 60)
    print("TEST 6: Edge Cases")
    print("=" * 60)

    # Single element
    print("Single [[5]]:", spiralOrder([[5]]))

    # N out of range
    matrix6 = [[1, 2], [3, 4]]
    print("N=10 in 2x2:", nthSpiralElement(matrix6, 10))

    # N = 1
    print("N=1:", nthSpiralElement(matrix6, 1))

    print("\n" + "=" * 60)
    print("TEST 7: Generate Spiral Matrix")
    print("=" * 60)

    n = 4
    spiral_matrix = generateSpiralMatrix(n)
    print("Generated " + str(n) + "x" + str(n) + " spiral matrix:")
    for row in spiral_matrix:
        print(" ", row)

    print("\n" + "=" * 60)
    print("TEST 8: Find Position")
    print("=" * 60)

    matrix8 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Matrix:")
    for row in matrix8:
        print(" ", row)
    print("Spiral:", spiralOrder(matrix8))
    print()

    for target in [1, 5, 9, 8]:
        pos = findPositionInSpiral(matrix8, target)
        print("Position of", target, ":", pos)

    print("\n" + "=" * 60)
    print("TEST 9: All Nth Elements")
    print("=" * 60)

    matrix9 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Matrix:")
    for row in matrix9:
        print(" ", row)
    print()

    total = len(matrix9) * len(matrix9[0])
    print("All Nth elements:")
    for n in range(1, total + 1):
        result = nthSpiralElement(matrix9, n)
        print("  N=" + str(n) + ":", result)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Nth Element of Spiral Matrix

Spiral Order:
  1. Left to Right (top row)
  2. Top to Bottom (right column)
  3. Right to Left (bottom row)
  4. Bottom to Top (left column)
  5. Repeat for inner layers

Four Boundaries:
  - top: starting row
  - bottom: ending row
  - left: starting column
  - right: ending column

Algorithm:
  - Traverse in spiral order
  - Count elements
  - Return when count == N

Time: O(rows * cols)
Space: O(1)
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print("""
Matrix:
  1  2  3
  4  5  6
  7  8  9

Spiral Traversal:

  1 → 2 → 3
            ↓
  4 → 5    6
  ↑        ↓
  7 ← 8 ← 9

Order: [1, 2, 3, 6, 9, 8, 7, 4, 5]

Position Mapping:
  N=1: 1    N=6: 8
  N=2: 2    N=7: 7
  N=3: 3    N=8: 4
  N=4: 6    N=9: 5
  N=5: 9
    """)

    print("\n" + "=" * 60)
    print("SPIRAL DIRECTIONS")
    print("=" * 60)
    print("""
Layer 1 (outer):
  → Right: 1, 2, 3
  ↓ Down:  6, 9
  ← Left:  8, 7
  ↑ Up:    4

Layer 2 (inner):
  → Right: 5

Boundaries after each step:
  Initial:  top=0, bottom=2, left=0, right=2
  After →:  top=1
  After ↓:  right=1
  After ←:  bottom=1
  After ↑:  left=1

  Layer 2:  top=1, bottom=1, left=1, right=1
  After →:  Done!
    """)

