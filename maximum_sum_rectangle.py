#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def maxSumRectangle(matrix):
    """
    Find maximum sum rectangle in 2D matrix

    Using Kadane's Algorithm on columns
    Time: O(rows * cols²), Space: O(rows)
    """
    if not matrix or not matrix[0]:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')

    # Fix left column
    for left in range(cols):
        temp = [0] * rows

        # Fix right column
        for right in range(left, cols):
            # Add current column to temp
            for row in range(rows):
                temp[row] += matrix[row][right]

            # Apply Kadane on temp
            current_sum = kadane(temp)
            max_sum = max(max_sum, current_sum)

    return max_sum


def kadane(arr):
    """
    Standard Kadane's algorithm
    """
    max_sum = arr[0]
    current = arr[0]

    for i in range(1, len(arr)):
        current = max(arr[i], current + arr[i])
        max_sum = max(max_sum, current)

    return max_sum


def maxSumRectangle_with_coords(matrix):
    """
    Return max sum with rectangle coordinates
    """
    if not matrix or not matrix[0]:
        return 0, None

    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')

    final_left = 0
    final_right = 0
    final_top = 0
    final_bottom = 0

    for left in range(cols):
        temp = [0] * rows

        for right in range(left, cols):
            for row in range(rows):
                temp[row] += matrix[row][right]

            # Kadane with position tracking
            current_max = temp[0]
            max_ending = temp[0]
            top = 0
            temp_top = 0
            bottom = 0

            for i in range(1, rows):
                if temp[i] > max_ending + temp[i]:
                    max_ending = temp[i]
                    temp_top = i
                else:
                    max_ending += temp[i]

                if max_ending > current_max:
                    current_max = max_ending
                    top = temp_top
                    bottom = i

            if current_max > max_sum:
                max_sum = current_max
                final_left = left
                final_right = right
                final_top = top
                final_bottom = bottom

    coords = (final_top, final_left, final_bottom, final_right)
    return max_sum, coords


def maxSumRectangle_detailed(matrix):
    """
    Step-by-step explanation
    """
    print("Maximum Sum Rectangle")
    print("=" * 60)
    print("Matrix:")
    for row in matrix:
        print(" ", row)
    print()

    if not matrix or not matrix[0]:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')

    print("Algorithm: Fix columns, apply Kadane on rows")
    print("-" * 60)

    for left in range(cols):
        temp = [0] * rows

        for right in range(left, cols):
            # Add column
            for row in range(rows):
                temp[row] += matrix[row][right]

            print("Columns [" + str(left) + "," + str(right) + "]:", temp, end=" ")

            # Kadane
            current = kadane(temp)
            print("Kadane:", current, end="")

            if current > max_sum:
                max_sum = current
                print(" ← MAX")
            else:
                print()

    print()
    print("=" * 60)
    print("Maximum Sum:", max_sum)

    return max_sum


def visualize_result(matrix):
    """
    Visual representation of result
    """
    print("Visualization")
    print("=" * 60)

    max_sum, coords = maxSumRectangle_with_coords(matrix)
    top, left, bottom, right = coords

    print("Matrix:")
    for row in matrix:
        print(" ", row)
    print()

    print("Maximum Sum:", max_sum)
    print("Rectangle: rows [" + str(top) + "," + str(bottom) + "]", 
          "cols [" + str(left) + "," + str(right) + "]")
    print()

    # Extract rectangle
    print("Selected Rectangle:")
    rect_sum = 0
    for i in range(top, bottom + 1):
        row_vals = []
        for j in range(left, right + 1):
            row_vals.append(matrix[i][j])
            rect_sum += matrix[i][j]
        print(" ", row_vals)

    print("Sum:", rect_sum)
    print()

    # Visual grid
    print("Visual (X = selected):")
    for i in range(len(matrix)):
        line = "  "
        for j in range(len(matrix[0])):
            if top <= i <= bottom and left <= j <= right:
                line += "[X] "
            else:
                line += " .  "
        print(line)

    return max_sum


def maxSumRectangle_bruteforce(matrix):
    """
    Brute force approach
    Time: O(rows² * cols² * rows * cols)
    """
    if not matrix or not matrix[0]:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')

    for top in range(rows):
        for bottom in range(top, rows):
            for left in range(cols):
                for right in range(left, cols):
                    current_sum = 0
                    for i in range(top, bottom + 1):
                        for j in range(left, right + 1):
                            current_sum += matrix[i][j]
                    max_sum = max(max_sum, current_sum)

    return max_sum


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Example")
    print("=" * 60)

    matrix1 = [
        [1, 2, -1, -4, -20],
        [-8, -3, 4, 2, 1],
        [3, 8, 10, 1, 3],
        [-4, -1, 1, 7, -6]
    ]

    print("Matrix:")
    for row in matrix1:
        print(" ", row)
    print()

    result = maxSumRectangle(matrix1)
    print("Maximum Sum:", result)
    print("Expected: 29")

    print("\n" + "=" * 60)
    print("TEST 2: With Coordinates")
    print("=" * 60)

    max_sum, coords = maxSumRectangle_with_coords(matrix1)
    print("Maximum Sum:", max_sum)
    print("Coordinates (top, left, bottom, right):", coords)

    top, left, bottom, right = coords
    print("\nSelected rectangle:")
    for i in range(top, bottom + 1):
        print(" ", matrix1[i][left:right + 1])

    print("\n" + "=" * 60)
    print("TEST 3: Detailed Explanation")
    print("=" * 60)

    matrix3 = [
        [1, 2, -1],
        [-3, 4, 5],
        [2, -1, 3]
    ]

    maxSumRectangle_detailed(matrix3)

    print("\n" + "=" * 60)
    print("TEST 4: Visualization")
    print("=" * 60)

    matrix4 = [
        [1, 2, -1],
        [-3, 4, 5],
        [2, -1, 3]
    ]

    visualize_result(matrix4)

    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)

    # Single element
    print("Single [[5]]:", maxSumRectangle([[5]]))

    # All positive
    print("All positive [[1,2],[3,4]]:", 
          maxSumRectangle([[1, 2], [3, 4]]))

    # All negative
    print("All negative [[-1,-2],[-3,-4]]:", 
          maxSumRectangle([[-1, -2], [-3, -4]]))

    # Single row
    print("Single row [[1,-2,3]]:", 
          maxSumRectangle([[1, -2, 3]]))

    # Single column
    print("Single column [[1],[2],[3]]:", 
          maxSumRectangle([[1], [2], [3]]))

    print("\n" + "=" * 60)
    print("TEST 6: Compare Methods")
    print("=" * 60)

    matrix6 = [
        [1, 2, -1],
        [-3, 4, 5],
        [2, -1, 3]
    ]

    result_kadane = maxSumRectangle(matrix6)
    result_brute = maxSumRectangle_bruteforce(matrix6)

    print("Kadane method:", result_kadane)
    print("Brute force:  ", result_brute)
    print("Match:", result_kadane == result_brute)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Maximum Sum Rectangle

Problem: Find rectangle with maximum sum in 2D matrix

Algorithm:
  1. Fix left column (L)
  2. Fix right column (R)  
  3. Compress columns L to R into 1D array
  4. Apply Kadane on 1D array
  5. Track maximum

Time: O(rows * cols²)
Space: O(rows)

Key: Convert 2D problem to 1D
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print("""
Matrix:
  1   2  -1
 -3   4   5
  2  -1   3

Fix columns [1, 2]:
  Row 0: 2 + (-1) = 1
  Row 1: 4 + 5 = 9
  Row 2: -1 + 3 = 2

1D array: [1, 9, 2]

Kadane([1, 9, 2]) = 12

This represents rectangle:
  rows [0-2], cols [1-2]

    2  -1
    4   5
   -1   3

Sum = 12 ✓
    """)

