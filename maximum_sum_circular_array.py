#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def maxCircularSum(arr):
    """
    Maximum Sum Circular Subarray

    Key Insight:
    - Case 1: Max subarray is in middle (normal Kadane)
    - Case 2: Max subarray wraps around (total - min subarray)

    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0

    n = len(arr)

    # Case 1: Normal Kadane (max subarray)
    max_sum = arr[0]
    current_max = arr[0]

    # Case 2: Find min subarray
    min_sum = arr[0]
    current_min = arr[0]

    total = arr[0]

    for i in range(1, n):
        total += arr[i]

        # Max subarray
        current_max = max(arr[i], current_max + arr[i])
        max_sum = max(max_sum, current_max)

        # Min subarray
        current_min = min(arr[i], current_min + arr[i])
        min_sum = min(min_sum, current_min)

    # If all elements are negative
    if max_sum < 0:
        return max_sum

    # Return max of both cases
    return max(max_sum, total - min_sum)


def maxCircularSum_detailed(arr):
    """
    Step-by-step explanation
    """
    print("Maximum Circular Subarray Sum")
    print("=" * 60)
    print("Array:", arr)
    print()

    if not arr:
        return 0

    n = len(arr)

    print("Two Cases:")
    print("Case 1: Max subarray in middle (no wrap)")
    print("Case 2: Max subarray wraps around")
    print()

    # Case 1: Normal Kadane
    print("CASE 1: Normal Kadane")
    print("-" * 40)

    max_sum = arr[0]
    current_max = arr[0]
    max_start, max_end = 0, 0
    temp_start = 0

    for i in range(1, n):
        if arr[i] > current_max + arr[i]:
            current_max = arr[i]
            temp_start = i
        else:
            current_max = current_max + arr[i]

        if current_max > max_sum:
            max_sum = current_max
            max_start = temp_start
            max_end = i

    print("Max subarray:", arr[max_start:max_end + 1])
    print("Max sum:", max_sum)
    print()

    # Case 2: Circular (total - min)
    print("CASE 2: Circular (Total - Min Subarray)")
    print("-" * 40)

    total = sum(arr)
    print("Total sum:", total)

    min_sum = arr[0]
    current_min = arr[0]

    for i in range(1, n):
        current_min = min(arr[i], current_min + arr[i])
        min_sum = min(min_sum, current_min)

    print("Min subarray sum:", min_sum)
    print("Circular sum:", total, "-", min_sum, "=", total - min_sum)
    print()

    # Result
    print("=" * 60)

    if max_sum < 0:
        print("All negative, return max element:", max_sum)
        return max_sum

    result = max(max_sum, total - min_sum)

    print("Case 1 (Normal):", max_sum)
    print("Case 2 (Circular):", total - min_sum)
    print("Maximum:", result)

    return result


def maxCircularSum_visual(arr):
    """
    Visual representation
    """
    print("Circular Array Visualization")
    print("=" * 60)
    print("Array:", arr)
    print()

    # Show circular nature
    print("Circular representation:")
    print("  " + " -> ".join(map(str, arr)) + " -> " + str(arr[0]) + " ...")
    print()

    result = maxCircularSum(arr)

    # Find the subarray
    n = len(arr)
    total = sum(arr)

    # Normal max
    max_sum = arr[0]
    current_max = arr[0]

    for i in range(1, n):
        current_max = max(arr[i], current_max + arr[i])
        max_sum = max(max_sum, current_max)

    # Circular max
    min_sum = arr[0]
    current_min = arr[0]

    for i in range(1, n):
        current_min = min(arr[i], current_min + arr[i])
        min_sum = min(min_sum, current_min)

    circular_max = total - min_sum

    print("Normal max subarray sum:", max_sum)
    print("Circular max sum:", circular_max)
    print()

    if max_sum >= circular_max:
        print("Best: Normal subarray")
    else:
        print("Best: Circular subarray (wraps around)")

    print("Answer:", result)

    return result


def maxCircularSum_bruteforce(arr):
    """
    Brute Force: Check all circular subarrays
    Time: O(n²), Space: O(1)
    """
    if not arr:
        return 0

    n = len(arr)
    max_sum = arr[0]

    for start in range(n):
        current_sum = 0

        for length in range(1, n + 1):
            idx = (start + length - 1) % n
            current_sum += arr[idx]
            max_sum = max(max_sum, current_sum)

    return max_sum


def findMaxCircularSubarray(arr):
    """
    Return max sum with start and end indices
    """
    if not arr:
        return 0, -1, -1

    n = len(arr)

    # Case 1: Normal Kadane
    max_sum = arr[0]
    current_max = arr[0]
    max_start = max_end = 0
    temp_start = 0

    for i in range(1, n):
        if arr[i] > current_max + arr[i]:
            current_max = arr[i]
            temp_start = i
        else:
            current_max += arr[i]

        if current_max > max_sum:
            max_sum = current_max
            max_start = temp_start
            max_end = i

    # Case 2: Circular
    total = sum(arr)

    min_sum = arr[0]
    current_min = arr[0]
    min_start = min_end = 0
    temp_start = 0

    for i in range(1, n):
        if arr[i] < current_min + arr[i]:
            current_min = arr[i]
            temp_start = i
        else:
            current_min += arr[i]

        if current_min < min_sum:
            min_sum = current_min
            min_start = temp_start
            min_end = i

    circular_sum = total - min_sum

    # All negative
    if max_sum < 0:
        return max_sum, max_start, max_end

    if max_sum >= circular_sum:
        return max_sum, max_start, max_end
    else:
        # Circular case: subarray is from min_end+1 to min_start-1
        circ_start = (min_end + 1) % n
        circ_end = (min_start - 1 + n) % n
        return circular_sum, circ_start, circ_end


def kadane(arr):
    """
    Standard Kadane for reference
    """
    max_sum = arr[0]
    current = arr[0]

    for i in range(1, len(arr)):
        current = max(arr[i], current + arr[i])
        max_sum = max(max_sum, current)

    return max_sum


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ([1, -2, 3, -2], 3, "Normal: [3]"),
        ([5, -3, 5], 10, "Circular: [5, 5]"),
        ([3, -1, 2, -1], 4, "Normal: [3, -1, 2]"),
        ([3, -2, 2, -3], 3, "Normal: [3]"),
        ([-2, -3, -1], -1, "All negative: [-1]"),
    ]

    for arr, expected, desc in test_cases:
        result = maxCircularSum(arr)
        status = "✅" if result == expected else "❌"
        print(status, arr, "→", result, "-", desc)

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    arr2 = [5, -3, 5]
    maxCircularSum_detailed(arr2)

    print("\n" + "=" * 60)
    print("TEST 3: Visual Representation")
    print("=" * 60)

    arr3 = [8, -1, -3, 8]
    maxCircularSum_visual(arr3)

    print("\n" + "=" * 60)
    print("TEST 4: With Indices")
    print("=" * 60)

    arr4 = [5, -3, 5]
    max_sum, start, end = findMaxCircularSubarray(arr4)

    print("Array:", arr4)
    print("Max sum:", max_sum)
    print("Start index:", start)
    print("End index:", end)

    if start <= end:
        print("Subarray:", arr4[start:end + 1])
    else:
        print("Subarray wraps:", arr4[start:] + arr4[:end + 1])

    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)

    edge_cases = [
        ([5], 5, "Single element"),
        ([-5], -5, "Single negative"),
        ([1, 2, 3], 6, "All positive"),
        ([-1, -2, -3], -1, "All negative"),
        ([0, 0, 0], 0, "All zeros"),
        ([1, -1, 1, -1], 2, "Alternating"),
    ]

    for arr, expected, desc in edge_cases:
        result = maxCircularSum(arr)
        status = "✅" if result == expected else "❌"
        print(status, desc.ljust(20), ":", arr, "→", result)

    print("\n" + "=" * 60)
    print("TEST 6: Compare with Brute Force")
    print("=" * 60)

    import random

    for _ in range(5):
        arr = [random.randint(-10, 10) for _ in range(6)]

        result_optimal = maxCircularSum(arr)
        result_brute = maxCircularSum_bruteforce(arr)

        status = "✅" if result_optimal == result_brute else "❌"
        print(status, arr, "→", result_optimal)

    print("\n" + "=" * 60)
    print("TEST 7: Normal Kadane vs Circular")
    print("=" * 60)

    comparisons = [
        [1, -2, 3, -2],      # Normal wins
        [5, -3, 5],          # Circular wins
        [-2, 4, -1, 4, -1],  # Circular wins
        [1, 2, 3, 4, 5],     # Same (all positive)
    ]

    for arr in comparisons:
        normal = kadane(arr)
        circular = maxCircularSum(arr)

        winner = "Normal" if circular == normal else "Circular"
        print(arr)
        print("  Normal:", normal, "| Circular:", circular, "| Winner:", winner)
        print()

    print("=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Maximum Circular Subarray Sum

Problem: Find max subarray sum where array is circular
         (last element connects to first)

Key Insight:
  Case 1: Max subarray in middle (normal)
          [... | MAX_SUBARRAY | ...]

  Case 2: Max subarray wraps around
          [MAX | ... min ... | MAX]
          = Total - Min_Subarray

Formula:
  Answer = max(
      max_subarray,           # Case 1: Kadane
      total - min_subarray    # Case 2: Circular
  )

Edge Case:
  If all negative, return max element

Time: O(n)
Space: O(1)

LeetCode #918: Maximum Sum Circular Subarray
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXPLANATION")
    print("=" * 60)
    print("""
Example: [5, -3, 5]

Case 1: Normal Kadane
  [5, -3, 5]
   ^------^  Sum = 5 + (-3) + 5 = 7

  Best normal = 7

Case 2: Circular
  Array is circular: ... 5 -> -3 -> 5 -> 5 -> -3 -> ...

  Total = 5 + (-3) + 5 = 7
  Min subarray = -3

  Circular sum = 7 - (-3) = 10

  This means: [5] + [5] = 10 (wrapping around)

       [5, -3, 5]
        ^      ^
        |______|  Wrap around, skip -3

Answer: max(7, 10) = 10

Visual:
        ┌──────────────┐
        │              ▼
    [ 5 | -3 | 5 ]
      ▲         │
      └─────────┘

    Take both 5s, skip -3
    Sum = 10
    """)

    print("\n" + "=" * 60)
    print("WHY TOTAL - MIN WORKS")
    print("=" * 60)
    print("""
Circular array:
[A | B | C | D | E]
 ↑_________________↑ (wraps)

If max wraps around:
[MAX_PART | MIN_SUBARRAY | MAX_PART]

Then:
Circular_Sum = Total - Min_Subarray

Because:
- We want elements NOT in min subarray
- Those elements = Total - Min_Subarray

Example: [8, -1, -3, 8]
Total = 12
Min subarray = [-1, -3] = -4
Circular = 12 - (-4) = 16

This equals: [8] + [8] = 16 (wrapping)
    """)

