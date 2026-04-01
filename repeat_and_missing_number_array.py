#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def findRepeatMissing(arr):
    """
    Find the repeating and missing number in array
    Array contains numbers from 1 to n with one repeated and one missing

    Using Math Formula (Optimal)
    Time: O(n), Space: O(1)
    """
    n = len(arr)

    # Sum of first n natural numbers: n(n+1)/2
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(arr)

    # Sum of squares: n(n+1)(2n+1)/6
    expected_sum_sq = n * (n + 1) * (2 * n + 1) // 6
    actual_sum_sq = sum(x * x for x in arr)

    # Let repeat = R, missing = M
    # R - M = actual_sum - expected_sum
    # R² - M² = actual_sum_sq - expected_sum_sq
    # R² - M² = (R + M)(R - M)
    # R + M = (R² - M²) / (R - M)

    diff = actual_sum - expected_sum  # R - M
    sum_diff_sq = actual_sum_sq - expected_sum_sq  # R² - M²

    total = sum_diff_sq // diff  # R + M

    repeat = (diff + total) // 2
    missing = total - repeat

    return repeat, missing


def findRepeatMissing_detailed(arr):
    """
    Step-by-step explanation
    """
    print("Find Repeat and Missing Number")
    print("=" * 60)
    print("Array:", arr)
    print("n =", len(arr))
    print()

    n = len(arr)

    # Expected values
    expected_sum = n * (n + 1) // 2
    expected_sum_sq = n * (n + 1) * (2 * n + 1) // 6

    print("Step 1: Calculate Expected Values")
    print("-" * 40)
    print("Expected sum (1 to n):", expected_sum)
    print("Expected sum of squares:", expected_sum_sq)
    print()

    # Actual values
    actual_sum = sum(arr)
    actual_sum_sq = sum(x * x for x in arr)

    print("Step 2: Calculate Actual Values")
    print("-" * 40)
    print("Actual sum:", actual_sum)
    print("Actual sum of squares:", actual_sum_sq)
    print()

    # Differences
    diff = actual_sum - expected_sum
    sum_diff_sq = actual_sum_sq - expected_sum_sq

    print("Step 3: Find Differences")
    print("-" * 40)
    print("R - M =", diff)
    print("R² - M² =", sum_diff_sq)
    print()

    # Solve equations
    print("Step 4: Solve Equations")
    print("-" * 40)
    print("R² - M² = (R + M)(R - M)")
    print("R + M = (R² - M²) / (R - M)")

    total = sum_diff_sq // diff
    print("R + M =", total)
    print()

    # Find R and M
    repeat = (diff + total) // 2
    missing = total - repeat

    print("From equations:")
    print("  R - M =", diff)
    print("  R + M =", total)
    print()
    print("Solving: R =", repeat, ", M =", missing)

    print()
    print("=" * 60)
    print("Repeat:", repeat)
    print("Missing:", missing)

    return repeat, missing


def findRepeatMissing_xor(arr):
    """
    Using XOR approach
    Time: O(n), Space: O(1)
    """
    n = len(arr)
    xor_all = 0

    # XOR of array elements
    for num in arr:
        xor_all ^= num

    # XOR with 1 to n
    for i in range(1, n + 1):
        xor_all ^= i

    # xor_all = R XOR M

    # Find rightmost set bit
    set_bit = xor_all & ~(xor_all - 1)

    x = 0  # Numbers with set bit
    y = 0  # Numbers without set bit

    # Divide into two groups
    for num in arr:
        if num & set_bit:
            x ^= num
        else:
            y ^= num

    for i in range(1, n + 1):
        if i & set_bit:
            x ^= i
        else:
            y ^= i

    # One of x or y is repeat, other is missing
    # Check which is which
    if arr.count(x) == 2:
        return x, y  # x is repeat, y is missing
    else:
        return y, x  # y is repeat, x is missing


def findRepeatMissing_hashmap(arr):
    """
    Using HashMap/Counter
    Time: O(n), Space: O(n)
    """
    from collections import Counter

    n = len(arr)
    count = Counter(arr)

    repeat = None
    missing = None

    for i in range(1, n + 1):
        if count[i] == 2:
            repeat = i
        elif count[i] == 0:
            missing = i

    return repeat, missing


def findRepeatMissing_array(arr):
    """
    Using frequency array
    Time: O(n), Space: O(n)
    """
    n = len(arr)
    freq = [0] * (n + 1)

    # Count frequencies
    for num in arr:
        freq[num] += 1

    repeat = None
    missing = None

    for i in range(1, n + 1):
        if freq[i] == 2:
            repeat = i
        elif freq[i] == 0:
            missing = i

    return repeat, missing


def findRepeatMissing_sorting(arr):
    """
    Using Sorting
    Time: O(n log n), Space: O(1)
    """
    arr_sorted = sorted(arr)
    n = len(arr)

    repeat = None
    missing = None

    # Find repeat
    for i in range(1, n):
        if arr_sorted[i] == arr_sorted[i - 1]:
            repeat = arr_sorted[i]
            break

    # Find missing
    for i in range(1, n + 1):
        if i not in arr_sorted:
            missing = i
            break

    return repeat, missing


def visualizeMethod(arr):
    """
    Visual representation of math method
    """
    print("Math Method Visualization")
    print("=" * 60)
    print("Array:", arr)
    print()

    n = len(arr)

    print("Given: Array has numbers 1 to", n)
    print("One number repeats, one is missing")
    print()

    expected_sum = n * (n + 1) // 2
    actual_sum = sum(arr)

    print("Expected sum (1+2+...+" + str(n) + "):", expected_sum)
    print("Actual sum:", actual_sum)
    print("Difference:", actual_sum - expected_sum)
    print()

    print("If R = repeat, M = missing:")
    print("  R - M =", actual_sum - expected_sum)
    print()

    expected_sum_sq = n * (n + 1) * (2 * n + 1) // 6
    actual_sum_sq = sum(x * x for x in arr)

    print("Expected sum of squares:", expected_sum_sq)
    print("Actual sum of squares:", actual_sum_sq)
    print()

    sum_diff_sq = actual_sum_sq - expected_sum_sq
    diff = actual_sum - expected_sum

    print("R² - M² =", sum_diff_sq)
    print("(R - M)(R + M) =", sum_diff_sq)
    print()

    total = sum_diff_sq // diff
    print("R + M = " + str(sum_diff_sq) + " / " + str(diff) + " =", total)
    print()

    repeat = (diff + total) // 2
    missing = total - repeat

    print("Solving system:")
    print("  R - M =", diff)
    print("  R + M =", total)
    print("  ----")
    print("  2R =", diff + total)
    print("  R =", repeat)
    print("  M =", missing)

    return repeat, missing


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ([1, 2, 3, 4, 5, 5], 5, 6),
        ([3, 1, 2, 5, 3], 3, 4),
        ([4, 3, 6, 2, 1, 1], 1, 5),
        ([1, 1], 1, 2),
    ]

    for arr, exp_repeat, exp_miss in test_cases:
        repeat, missing = findRepeatMissing(arr)
        status = "✅" if repeat == exp_repeat and missing == exp_miss else "❌"
        print(status, arr, "→ Repeat:", repeat, "Missing:", missing)

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    findRepeatMissing_detailed([3, 1, 2, 5, 3])

    print("\n" + "=" * 60)
    print("TEST 3: Compare All Methods")
    print("=" * 60)

    arr3 = [4, 3, 6, 2, 1, 1]
    print("Array:", arr3)
    print()

    r1, m1 = findRepeatMissing(arr3)
    print("Math Formula:  Repeat =", r1, ", Missing =", m1)

    r2, m2 = findRepeatMissing_xor(arr3)
    print("XOR Method:    Repeat =", r2, ", Missing =", m2)

    r3, m3 = findRepeatMissing_hashmap(arr3)
    print("HashMap:       Repeat =", r3, ", Missing =", m3)

    r4, m4 = findRepeatMissing_array(arr3)
    print("Freq Array:    Repeat =", r4, ", Missing =", m4)

    r5, m5 = findRepeatMissing_sorting(arr3)
    print("Sorting:       Repeat =", r5, ", Missing =", m5)

    print("\n" + "=" * 60)
    print("TEST 4: Visualization")
    print("=" * 60)

    visualizeMethod([3, 1, 2, 5, 3])

    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)

    edge_cases = [
        ([1, 1], "Smallest case"),
        ([1, 2, 2], "n = 3"),
        ([5, 4, 3, 2, 1, 1], "Reverse order"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10], "Larger array"),
    ]

    for arr, desc in edge_cases:
        repeat, missing = findRepeatMissing(arr)
        print(desc + ":")
        print("  Array:", arr)
        print("  Repeat:", repeat, ", Missing:", missing)
        print()

    print("\n" + "=" * 60)
    print("TEST 6: Performance")
    print("=" * 60)

    import time

    # Create large test array
    n = 10000
    test_arr = list(range(1, n + 1))
    test_arr[5000] = 100  # Make 100 repeat, 5001 missing

    print("Array size:", n)
    print()

    # Math method
    start = time.time()
    r, m = findRepeatMissing(test_arr)
    time_math = time.time() - start

    # HashMap method
    start = time.time()
    r2, m2 = findRepeatMissing_hashmap(test_arr)
    time_hash = time.time() - start

    print("Math Formula:  ", round(time_math, 6), "s")
    print("HashMap:       ", round(time_hash, 6), "s")
    print()
    print("Result: Repeat =", r, ", Missing =", m)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Find Repeat and Missing Number

Problem: Array of size n contains numbers 1 to n
         One number repeats, one is missing
         Find both

Method 1: Math Formula (Optimal) ⭐
  - Use sum and sum of squares
  - R - M = actual_sum - expected_sum
  - R + M = (R² - M²) / (R - M)
  - Solve to get R and M
  Time: O(n), Space: O(1)

Method 2: XOR
  - XOR properties
  - Bit manipulation
  Time: O(n), Space: O(1)

Method 3: HashMap/Counter
  - Count frequency
  - freq[R] = 2, freq[M] = 0
  Time: O(n), Space: O(n)

Method 4: Sorting
  - Sort and find
  Time: O(n log n), Space: O(1)

Best: Math Formula (O(n) time, O(1) space)
    """)

    print("\n" + "=" * 60)
    print("MATH FORMULA EXPLAINED")
    print("=" * 60)
    print("""
Given: Numbers 1 to n, one repeats (R), one missing (M)

Equations:
  1. R - M = actual_sum - expected_sum
  2. R² - M² = actual_sum_sq - expected_sum_sq

From equation 2:
  R² - M² = (R + M)(R - M)
  R + M = (R² - M²) / (R - M)

Now we have:
  R - M = diff
  R + M = total

Solving:
  2R = diff + total
  R = (diff + total) / 2
  M = total - R

Example: [3, 1, 2, 5, 3]
  n = 5
  Expected sum = 15, Actual = 14
  R - M = -1

  Expected sum² = 55, Actual = 48
  R² - M² = -7
  R + M = -7 / -1 = 7

  R = (-1 + 7) / 2 = 3
  M = 7 - 3 = 4
    """)

