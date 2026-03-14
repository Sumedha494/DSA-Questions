#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def kadane(arr):
    """
    Kadane's Algorithm - Maximum Subarray Sum
    Time Complexity: O(n)
    Space Complexity: O(1)

    Ek hi pass mein maximum sum subarray dhundho
    """
    if not arr:
        return 0

    max_sum = arr[0]  # Global maximum
    current_sum = arr[0]  # Current subarray sum

    for i in range(1, len(arr)):
        # Ya toh current element ko add karo, ya naya subarray start karo
        current_sum = max(arr[i], current_sum + arr[i])

        # Global maximum update karo
        max_sum = max(max_sum, current_sum)

    return max_sum


def kadane_detailed(arr):
    """
    Step-by-step explanation ke saath
    """
    print("Finding Maximum Subarray Sum using Kadane's Algorithm")
    print("=" * 70)
    print("Array:", arr)
    print()

    if not arr:
        return 0

    max_sum = arr[0]
    current_sum = arr[0]

    print("Initial: max_sum =", max_sum, ", current_sum =", current_sum)
    print("-" * 70)

    for i in range(1, len(arr)):
        old_current = current_sum

        # Decision: extend ya new start?
        if arr[i] > current_sum + arr[i]:
            current_sum = arr[i]
            decision = "Start new subarray"
        else:
            current_sum = current_sum + arr[i]
            decision = "Extend current subarray"

        print("Index", i, ": Element =", arr[i])
        print("  Options: ", arr[i], "vs", old_current, "+", arr[i], "=", old_current + arr[i])
        print("  Decision:", decision)
        print("  current_sum:", current_sum)

        # Update global max
        if current_sum > max_sum:
            max_sum = current_sum
            print("  ✓ New max_sum found:", max_sum)

        print()

    print("=" * 70)
    print("Maximum Subarray Sum:", max_sum)
    return max_sum


def kadane_with_indices(arr):
    """
    Maximum sum ke saath subarray ki start aur end indices bhi return karo
    """
    if not arr:
        return 0, -1, -1

    max_sum = arr[0]
    current_sum = arr[0]

    start = 0  # Current subarray start
    end = 0    # Maximum subarray end
    temp_start = 0  # Temporary start for new subarray

    for i in range(1, len(arr)):
        # Agar naya start better hai
        if arr[i] > current_sum + arr[i]:
            current_sum = arr[i]
            temp_start = i
        else:
            current_sum = current_sum + arr[i]

        # Global maximum update
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i

    return max_sum, start, end


def kadane_print_subarray(arr):
    """
    Maximum sum subarray ko print karo
    """
    max_sum, start, end = kadane_with_indices(arr)

    print("Array:", arr)
    print("Maximum Subarray Sum:", max_sum)
    print("Start Index:", start)
    print("End Index:", end)
    print("Subarray:", arr[start:end + 1])

    return max_sum


def kadane_all_negative(arr):
    """
    Handle case when all elements are negative
    Standard kadane works, but this is explicit version
    """
    if not arr:
        return 0

    # Sabhi negative hain to largest element hi answer hai
    if all(x < 0 for x in arr):
        return max(arr)

    # Standard kadane
    return kadane(arr)


def kadane_circular(arr):
    """
    Circular Array mein Maximum Subarray Sum

    Circular ka matlab: last element ke baad first element aa sakta hai

    Approach:
    1. Normal kadane
    2. Total sum - minimum subarray sum (circular case)
    3. Maximum of both
    """
    if not arr:
        return 0

    # Case 1: Normal kadane
    max_kadane = kadane(arr)

    # Case 2: Circular sum
    # Total sum - minimum subarray sum
    total_sum = sum(arr)

    # Minimum subarray sum (negate array and find max)
    arr_inverted = [-x for x in arr]
    max_inverted = kadane(arr_inverted)
    min_subarray = -max_inverted

    max_circular = total_sum - min_subarray

    # Edge case: Agar saare elements negative
    if max_circular == 0:
        return max_kadane

    return max(max_kadane, max_circular)


def kadane_2d(matrix):
    """
    2D Matrix mein Maximum Sum Rectangle

    Approach:
    1. Fix top and bottom rows
    2. Columns ko compress karke 1D array banao
    3. 1D array pe kadane lagao
    """
    if not matrix or not matrix[0]:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')

    # Fix top row
    for top in range(rows):
        # Array to store column sums
        temp = [0] * cols

        # Fix bottom row
        for bottom in range(top, rows):
            # Add current row to temp
            for col in range(cols):
                temp[col] += matrix[bottom][col]

            # Apply kadane on temp
            current_sum = kadane(temp)
            max_sum = max(max_sum, current_sum)

    return max_sum


def kadane_with_count(arr, k):
    """
    Exactly k elements ka maximum sum subarray
    Dynamic Programming approach
    """
    n = len(arr)
    if k > n or k <= 0:
        return 0

    # dp[i][j] = max sum using exactly j elements ending at i
    dp = [[float('-inf')] * (k + 1) for _ in range(n)]

    # Base case: 1 element
    for i in range(n):
        dp[i][1] = arr[i]

    # Fill dp table
    for i in range(1, n):
        for j in range(2, min(i + 2, k + 1)):
            # Include arr[i] in subarray
            dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + arr[i])

    # Maximum among all positions with exactly k elements
    max_sum = float('-inf')
    for i in range(k - 1, n):
        max_sum = max(max_sum, dp[i][k])

    return max_sum


def visualize_kadane(arr):
    """
    Visual representation of kadane's algorithm
    """
    print("Kadane's Algorithm Visualization")
    print("=" * 70)
    print("Array:", arr)
    print()

    if not arr:
        return 0

    max_sum = arr[0]
    current_sum = arr[0]
    max_start = 0
    max_end = 0
    temp_start = 0

    print("Step 0: Initial state")
    print("  current_sum =", current_sum, ", max_sum =", max_sum)
    print()

    for i in range(1, len(arr)):
        print("Step", i, ": Processing arr[" + str(i) + "] =", arr[i])

        if arr[i] > current_sum + arr[i]:
            current_sum = arr[i]
            temp_start = i
            print("  → Starting new subarray from index", i)
        else:
            current_sum = current_sum + arr[i]
            print("  → Extending current subarray")

        print("  current_sum =", current_sum)

        if current_sum > max_sum:
            max_sum = current_sum
            max_start = temp_start
            max_end = i
            print("  ✓ New maximum found!")
            print("  max_sum =", max_sum)
            print("  Subarray:", arr[max_start:max_end + 1])

        print()

    print("=" * 70)
    print("Final Answer:")
    print("Maximum Sum:", max_sum)
    print("Subarray:", arr[max_start:max_end + 1])
    print("Indices: [" + str(max_start) + ":" + str(max_end) + "]")


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEST 1: Basic Kadane's Algorithm")
    print("=" * 70)
    arr1 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print("Array:", arr1)
    result1 = kadane(arr1)
    print("Maximum Subarray Sum:", result1)
    print("Expected: 6 (subarray [4, -1, 2, 1])")

    print("\n" + "=" * 70)
    print("TEST 2: Detailed Explanation")
    print("=" * 70)
    arr2 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    kadane_detailed(arr2)

    print("\n" + "=" * 70)
    print("TEST 3: With Subarray Indices")
    print("=" * 70)
    arr3 = [1, 2, 3, -2, 5]
    kadane_print_subarray(arr3)

    print("\n" + "=" * 70)
    print("TEST 4: All Negative Numbers")
    print("=" * 70)
    arr4 = [-5, -2, -8, -1, -4]
    print("Array:", arr4)
    result4 = kadane(arr4)
    print("Maximum Sum:", result4)
    print("Note: When all negative, largest element is answer")

    print("\n" + "=" * 70)
    print("TEST 5: All Positive Numbers")
    print("=" * 70)
    arr5 = [1, 2, 3, 4, 5]
    print("Array:", arr5)
    result5 = kadane(arr5)
    print("Maximum Sum:", result5)
    print("Note: When all positive, sum of all elements is answer")

    print("\n" + "=" * 70)
    print("TEST 6: Circular Array")
    print("=" * 70)
    arr6 = [5, -3, 5]
    print("Array:", arr6)
    normal_sum = kadane(arr6)
    circular_sum = kadane_circular(arr6)
    print("Normal Kadane:", normal_sum)
    print("Circular Kadane:", circular_sum)
    print("Circular wins because [5, 5] wraps around")

    print("\n" + "=" * 70)
    print("TEST 7: 2D Matrix Maximum Rectangle")
    print("=" * 70)
    matrix = [
        [1, 2, -1, -4, -20],
        [-8, -3, 4, 2, 1],
        [3, 8, 10, 1, 3],
        [-4, -1, 1, 7, -6]
    ]
    print("Matrix:")
    for row in matrix:
        print(row)
    result7 = kadane_2d(matrix)
    print("Maximum Sum Rectangle:", result7)

    print("\n" + "=" * 70)
    print("TEST 8: Visualization")
    print("=" * 70)
    arr8 = [2, -1, 2, 3, -2, 4]
    visualize_kadane(arr8)

    print("\n" + "=" * 70)
    print("TEST 9: Edge Cases")
    print("=" * 70)

    edge_cases = [
        ([5], "Single positive"),
        ([-5], "Single negative"),
        ([0, 0, 0], "All zeros"),
        ([1, -1, 1, -1], "Alternating"),
        ([], "Empty array"),
    ]

    for arr, desc in edge_cases:
        result = kadane(arr) if arr else 0
        print(desc.ljust(20), ":", arr, "→", result)

    print("\n" + "=" * 70)
    print("TEST 10: LeetCode Style Problems")
    print("=" * 70)

    # Problem 1: Maximum Subarray
    arr10_1 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print("Problem: Maximum Subarray (LeetCode #53)")
    print("Array:", arr10_1)
    max_sum, start, end = kadane_with_indices(arr10_1)
    print("Answer:", max_sum)
    print("Subarray:", arr10_1[start:end + 1], "\n")

    # Problem 2: Maximum 

