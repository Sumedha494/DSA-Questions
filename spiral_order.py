#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def spiral_order(matrix):
    """
    Returns a list of elements in spiral order.
    Time: O(m*n)
    Space: O(1) (excluding output list)
    """
    if not matrix or not matrix[0]:
        return []

    result = []

    # Define boundaries
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # 1. Traverse Left to Right (Top Row)
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1

        # 2. Traverse Top to Bottom (Right Column)
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1

        # 3. Traverse Right to Left (Bottom Row)
        # Check if top <= bottom to prevent duplicate printing in single row case
        if top <= bottom:
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1

        # 4. Traverse Bottom to Top (Left Column)
        # Check if left <= right to prevent duplicate printing in single col case
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result

# --- Example Usage ---
matrix = [
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
]

print("Original Matrix:")
for row in matrix: print(row)

print("\nSpiral Order:")
print(spiral_order(matrix))


# In[ ]:


def spiral_order_recursive(matrix):
    """
    Recursive approach to spiral order.
    """
    result = []

    def traverse(top, bottom, left, right):
        # Base case: boundaries crossed
        if top > bottom or left > right:
            return

        # 1. Left to Right
        for i in range(left, right + 1):
            result.append(matrix[top][i])

        # 2. Top to Bottom
        for i in range(top + 1, bottom + 1):
            result.append(matrix[i][right])

        # 3. Right to Left (if not a single row)
        if top < bottom:
            for i in range(right - 1, left - 1, -1):
                result.append(matrix[bottom][i])

        # 4. Bottom to Top (if not a single col)
        if left < right:
            for i in range(bottom - 1, top, -1):
                result.append(matrix[i][left])

        # Recurse for inner matrix
        traverse(top + 1, bottom - 1, left + 1, right - 1)

    if matrix:
        traverse(0, len(matrix) - 1, 0, len(matrix[0]) - 1)

    return result

# --- Example ---
m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(spiral_order_recursive(m))
# Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]


# In[ ]:


def spiral_order_pythonic(matrix):
    result = []
    # Work on a copy if you don't want to destroy original
    matrix = [row[:] for row in matrix] 

    while matrix:
        # 1. Take the first row
        result += matrix.pop(0)

        # 2. Rotate the remaining matrix Counter-Clockwise
        if matrix and matrix[0]:
            matrix = list(zip(*matrix))[::-1]
            # Convert tuples back to lists (optional)
            matrix = [list(row) for row in matrix]

    return result

# --- Example ---
m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(spiral_order_pythonic(m))
# Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]

