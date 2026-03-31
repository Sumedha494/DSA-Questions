#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def findAllSubsquares(matrix, k):
    """
    Find all subsquares of size K x K in a matrix

    Time: O((n-k+1) * (m-k+1) * k²)
    Space: O(1) excluding output
    """
    if not matrix or not matrix[0]:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    if k > rows or k > cols:
        return []

    subsquares = []

    for i in range(rows - k + 1):
        for j in range(cols - k + 1):
            # Extract k x k subsquare
            square = []
            for r in range(i, i + k):
                row = []
                for c in range(j, j + k):
                    row.append(matrix[r][c])
                square.append(row)

            subsquares.append(square)

    return subsquares


def findAllSubsquares_with_positions(matrix, k):
    """
    Return subsquares with their top-left positions
    """
    if not matrix or not matrix[0]:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    if k > rows or k > cols:
        return []

    result = []

    for i in range(rows - k + 1):
        for j in range(cols - k + 1):
            square = []
            for r in range(i, i + k):
                row = matrix[r][j:j + k]
                square.append(row)

            result.append({
                'position': (i, j),
                'square': square
            })

    return result


def findAllSubsquares_detailed(matrix, k):
    """
    Step-by-step explanation
    """
    print("Find All Subsquares of Size K")
    print("=" * 50)
    print("Matrix:")
    printMatrix(matrix)
    print("K:", k)
    print()

    if not matrix or not matrix[0]:
        print("Empty matrix!")
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    print("Matrix size:", rows, "x", cols)
    print("Subsquare size:", k, "x", k)

    if k > rows or k > cols:
        print("K is larger than matrix!")
        return []

    total = (rows - k + 1) * (cols - k + 1)
    print("Total subsquares:", total)
    print()

    print("All Subsquares:")
    print("-" * 50)

    subsquares = []
    count = 0

    for i in range(rows - k + 1):
        for j in range(cols - k + 1):
            count += 1

            square = []
            for r in range(i, i + k):
                row = matrix[r][j:j + k]
                square.append(row)

            subsquares.append(square)

            print("Subsquare", count, "at position (" + str(i) + "," + str(j) + "):")
            printMatrix(square, indent=2)
            print()

    print("=" * 50)
    print("Total found:", len(subsquares))

    return subsquares


def printMatrix(matrix, indent=0):
    """
    Print matrix nicely
    """
    prefix = " " * indent
    for row in matrix:
        print(prefix, row)


def sumOfSubsquares(matrix, k):
    """
    Find sum of each K x K subsquare
    """
    if not matrix or not matrix[0]:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    if k > rows or k > cols:
        return []

    sums = []

    for i in range(rows - k + 1):
        row_sums = []
        for j in range(cols - k + 1):
            total = 0
            for r in range(i, i + k):
                for c in range(j, j + k):
                    total += matrix[r][c]
            row_sums.append(total)
        sums.append(row_sums)

    return sums


def maxSumSubsquare(matrix, k):
    """
    Find subsquare with maximum sum
    """
    if not matrix or not matrix[0]:
        return 0, None

    rows = len(matrix)
    cols = len(matrix[0])

    if k > rows or k > cols:
        return 0, None

    max_sum = float('-inf')
    max_pos = None
    max_square = None

    for i in range(rows - k + 1):
        for j in range(cols - k + 1):
            total = 0
            square = []

            for r in range(i, i + k):
                row = matrix[r][j:j + k]
                square.append(row)
                total += sum(row)

            if total > max_sum:
                max_sum = total
                max_pos = (i, j)
                max_square = square

    return max_sum, max_pos, max_square


def countSubsquares(matrix, k):
    """
    Count total number of K x K subsquares
    """
    if not matrix or not matrix[0]:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])

    if k > rows or k > cols:
        return 0

    return (rows - k + 1) * (cols - k + 1)


def visualizeSubsquares(matrix, k):
    """
    Visual representation
    """
    print("Subsquare Visualization")
    print("=" * 50)
    print("Matrix:")
    printMatrix(matrix)
    print()
    print("K =", k)
    print()

    rows = len(matrix)
    cols = len(matrix[0])

    if k > rows or k > cols:
        print("K too large!")
        return

    print("Subsquare positions:")
    print("-" * 40)

    count = 0
    for i in range(rows - k + 1):
        for j in range(cols - k + 1):
            count += 1
            print("Position", count, ": top-left at (" + str(i) + "," + str(j) + ")")

            # Show which elements are included
            elements = []
            for r in range(i, i + k):
                for c in range(j, j + k):
                    elements.append(matrix[r][c])
            print("  Elements:", elements)

    print()
    print("Total subsquares:", count)


def findSubsquaresWithCondition(matrix, k, condition):
    """
    Find subsquares that satisfy a condition
    condition is a function that takes a square and returns True/False
    """
    if not matrix or not matrix[0]:
        return []

    rows = len(matrix)
    cols = len(matrix[0])

    if k > rows or k > cols:
        return []

    result = []

    for i in range(rows - k + 1):
        for j in range(cols - k + 1):
            square = []
            for r in range(i, i + k):
                row = matrix[r][j:j + k]
                square.append(row)

            if condition(square):
                result.append({
                    'position': (i, j),
                    'square': square
                })

    return result


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 50)
    print("TEST 1: Basic Example")
    print("=" * 50)

    matrix1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    k1 = 2
    print("Matrix:")
    printMatrix(matrix1)
    print("K:", k1)
    print()

    subsquares = findAllSubsquares(matrix1, k1)
    print("Found", len(subsquares), "subsquares:")
    for i, sq in enumerate(subsquares, 1):
        print("Subsquare", str(i) + ":")
        printMatrix(sq, indent=2)
        print()

    print("\n" + "=" * 50)
    print("TEST 2: Detailed Explanation")
    print("=" * 50)

    matrix2 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    findAllSubsquares_detailed(matrix2, 2)

    print("\n" + "=" * 50)
    print("TEST 3: With Positions")
    print("=" * 50)

    matrix3 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]

    print("Matrix:")
    printMatrix(matrix3)
    print("K: 2")
    print()

    result = findAllSubsquares_with_positions(matrix3, 2)
    for item in result:
        pos = item['position']
        sq = item['square']
        print("Position", pos, ":")
        printMatrix(sq, indent=2)
        print()

    print("\n" + "=" * 50)
    print("TEST 4: Sum of Subsquares")
    print("=" * 50)

    matrix4 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Matrix:")
    printMatrix(matrix4)
    print()

    sums = sumOfSubsquares(matrix4, 2)
    print("Sums of 2x2 subsquares:")
    printMatrix(sums)

    print("\n" + "=" * 50)
    print("TEST 5: Maximum Sum Subsquare")
    print("=" * 50)

    matrix5 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Matrix:")
    printMatrix(matrix5)
    print()

    max_sum, pos, square = maxSumSubsquare(matrix5, 2)
    print("K: 2")
    print("Maximum sum:", max_sum)
    print("Position:", pos)
    print("Subsquare:")
    printMatrix(square, indent=2)

    print("\n" + "=" * 50)
    print("TEST 6: Count Subsquares")
    print("=" * 50)

    matrix6 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    print("Matrix: 4x4")
    for k in range(1, 5):
        count = countSubsquares(matrix6, k)
        print("K=" + str(k) + ": " + str(count) + " subsquares")

    print("\n" + "=" * 50)
    print("TEST 7: Visualization")
    print("=" * 50)

    matrix7 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    visualizeSubsquares(matrix7, 2)

    print("\n" + "=" * 50)
    print("TEST 8: Edge Cases")
    print("=" * 50)

    # K equals matrix size
    matrix8 = [[1, 2], [3, 4]]
    sq8 = findAllSubsquares(matrix8, 2)
    print("Matrix 2x2, K=2:", len(sq8), "subsquare")

    # K = 1
    sq9 = findAllSubsquares(matrix8, 1)
    print("Matrix 2x2, K=1:", len(sq9), "subsquares")

    # K larger than matrix
    sq10 = findAllSubsquares(matrix8, 3)
    print("Matrix 2x2, K=3:", len(sq10), "subsquares")

    print("\n" + "=" * 50)
    print("TEST 9: Large Matrix")
    print("=" * 50)

    matrix9 = [[i * 5 + j + 1 for j in range(5)] for i in range(5)]
    print("Matrix 5x5:")
    printMatrix(matrix9)
    print()

    for k in [2, 3, 4]:
        count = countSubsquares(matrix9, k)
        print("K=" + str(k) + ":", count, "subsquares")

    print("\n" + "=" * 50)
    print("TEST 10: Find with Condition")
    print("=" * 50)

    matrix10 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    print("Matrix:")
    printMatrix(matrix10)
    print()
    print("Finding 2x2 subsquares with sum > 20:")

    def sum_greater_than_20(square):
        total = sum(sum(row) for row in square)
        return total > 20

    result = findSubsquaresWithCondition(matrix10, 2, sum_greater_than_20)

    for item in result:
        pos = item['position']
        sq = item['square']
        total = sum(sum(row) for row in sq)
        print("Position", pos, ", Sum =", total)
        printMatrix(sq, indent=2)
        print()

    print("\n" + "=" * 50)
    print("ALGORITHM SUMMARY")
    print("=" * 50)
    print()
    print("Find All Subsquares of Size K")
    print()
    print("Formula:")
    print("  Total subsquares = (rows - k + 1) * (cols - k + 1)")
    print()
    print("Algorithm:")
    print("  1. Loop i from 0 to rows - k")
    print("  2. Loop j from 0 to cols - k")
    print("  3. Extract K x K square starting at (i, j)")
    print()
    print("Time: O((n-k+1) * (m-k+1) * k^2)")
    print("Space: O(k^2) per subsquare")

    print("\n" + "=" * 50)
    print("VISUAL EXAMPLE")
    print("=" * 50)
    print()
    print("Matrix (3x3):")
    print("  1  2  3")
    print("  4  5  6")
    print("  7  8  9")
    print()
    print("K = 2, find 2x2 subsquares:")
    print()
    print("Position (0,0):  Position (0,1):")
    print("  [1, 2]           [2, 3]")
    print("  [4, 5]           [5, 6]")
    print()
    print("Position (1,0):  Position (1,1):")
    print("  [4, 5]           [5, 6]")
    print("  [7, 8]           [8, 9]")
    print()
    print("Total: (3-2+1) * (3-2+1) = 2 * 2 = 4")

