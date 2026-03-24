#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def findTriplet(arr):
    """
    Find if sum of any two elements equals third element
    arr[i] + arr[j] = arr[k]

    Approach: Sort + Two Pointer
    Time: O(n²), Space: O(1)
    """
    if len(arr) < 3:
        return False

    arr.sort()

    # For each element as target (from end)
    for k in range(len(arr) - 1, 1, -1):
        target = arr[k]
        left = 0
        right = k - 1

        # Two pointer to find sum
        while left < right:
            current_sum = arr[left] + arr[right]

            if current_sum == target:
                return True
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return False


def findTriplet_with_indices(arr):
    """
    Return indices of triplet if exists
    """
    if len(arr) < 3:
        return None

    arr_indexed = [(val, i) for i, val in enumerate(arr)]
    arr_indexed.sort()

    for k in range(len(arr_indexed) - 1, 1, -1):
        target = arr_indexed[k][0]
        left = 0
        right = k - 1

        while left < right:
            current_sum = arr_indexed[left][0] + arr_indexed[right][0]

            if current_sum == target:
                return (arr_indexed[left][1], 
                       arr_indexed[right][1], 
                       arr_indexed[k][1])
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return None


def findTriplet_detailed(arr):
    """
    Step-by-step explanation
    """
    print("Find if Sum of Two = Third")
    print("=" * 60)
    print("Array:", arr)
    print()

    if len(arr) < 3:
        print("Array too small!")
        return False

    arr_copy = arr.copy()
    arr_copy.sort()

    print("Sorted:", arr_copy)
    print()
    print("Checking each element as target (from largest):")
    print("-" * 60)

    for k in range(len(arr_copy) - 1, 1, -1):
        target = arr_copy[k]
        left = 0
        right = k - 1

        print("\nTarget: arr[" + str(k) + "] = " + str(target))
        print("Finding if arr[i] + arr[j] = " + str(target))

        while left < right:
            current_sum = arr_copy[left] + arr_copy[right]

            print("  arr[" + str(left) + "]=" + str(arr_copy[left]), 
                  "+ arr[" + str(right) + "]=" + str(arr_copy[right]), 
                  "= " + str(current_sum), end=" ")

            if current_sum == target:
                print("✓ FOUND!")
                print()
                print("=" * 60)
                print("Result: " + str(arr_copy[left]) + " + " + 
                      str(arr_copy[right]) + " = " + str(target))
                return True
            elif current_sum < target:
                print("(too small, left++)")
                left += 1
            else:
                print("(too large, right--)")
                right -= 1

    print()
    print("=" * 60)
    print("No triplet found")
    return False


def findTriplet_hashset(arr):
    """
    Using HashSet
    Time: O(n²), Space: O(n)
    """
    if len(arr) < 3:
        return False

    arr_set = set(arr)

    # Check all pairs
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            sum_val = arr[i] + arr[j]

            # Check if sum exists in array
            if sum_val in arr_set:
                return True

    return False


def findAllTriplets(arr):
    """
    Find all triplets where a + b = c
    """
    if len(arr) < 3:
        return []

    arr.sort()
    result = []

    for k in range(len(arr) - 1, 1, -1):
        target = arr[k]
        left = 0
        right = k - 1

        while left < right:
            current_sum = arr[left] + arr[right]

            if current_sum == target:
                result.append((arr[left], arr[right], target))
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return result


def findTriplet_brute_force(arr):
    """
    Brute Force: Check all combinations
    Time: O(n³), Space: O(1)
    """
    n = len(arr)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j:
                    if arr[i] + arr[j] == arr[k]:
                        return True

    return False


def visualize_two_pointer(arr):
    """
    Visual representation of two pointer approach
    """
    print("Two Pointer Visualization")
    print("=" * 60)

    arr.sort()
    print("Sorted:", arr)
    print()

    for k in range(len(arr) - 1, 1, -1):
        target = arr[k]
        left = 0
        right = k - 1

        print("Target:", target)

        while left < right:
            # Visual pointers
            pointer_line = ""
            for i in range(len(arr)):
                if i == left:
                    pointer_line += " L  "
                elif i == right:
                    pointer_line += " R  "
                elif i == k:
                    pointer_line += " T  "
                else:
                    pointer_line += "    "

            print("Array:", arr)
            print("      ", pointer_line)

            current_sum = arr[left] + arr[right]

            print("Sum:", arr[left], "+", arr[right], "=", current_sum)

            if current_sum == target:
                print("✓ FOUND!", arr[left], "+", arr[right], "=", target)
                print()
                return True
            elif current_sum < target:
                print("Too small, move left →")
                left += 1
            else:
                print("Too large, move right ←")
                right -= 1

            print()

        print("-" * 40)

    return False


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ([5, 3, 4], True, "3 + 4 = 7? No, but 3 + 5 = 8? No"),
        ([1, 2, 3], True, "1 + 2 = 3"),
        ([5, 8, 3], True, "3 + 5 = 8"),
        ([1, 4, 6], False, "No triplet"),
        ([2, 3, 5], True, "2 + 3 = 5"),
    ]

    for arr, expected, desc in test_cases:
        result = findTriplet(arr.copy())
        status = "✅" if result == expected else "❌"
        print(status, arr, "→", result, "-", desc)

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    arr2 = [5, 8, 3, 1]
    findTriplet_detailed(arr2)

    print("\n" + "=" * 60)
    print("TEST 3: With Indices")
    print("=" * 60)

    arr3 = [5, 8, 3, 1]
    print("Array:", arr3)
    indices = findTriplet_with_indices(arr3)
    if indices:
        i, j, k = indices
        print("Found: arr[" + str(i) + "]=" + str(arr3[i]), 
              "+ arr[" + str(j) + "]=" + str(arr3[j]),
              "= arr[" + str(k) + "]=" + str(arr3[k]))
    else:
        print("No triplet found")

    print("\n" + "=" * 60)
    print("TEST 4: Find All Triplets")
    print("=" * 60)

    arr4 = [1, 2, 3, 4, 5, 6]
    print("Array:", arr4)
    all_triplets = findAllTriplets(arr4.copy())
    print("All triplets (a + b = c):")
    for a, b, c in all_triplets:
        print("  ", a, "+", b, "=", c)

    print("\n" + "=" * 60)
    print("TEST 5: Visualization")
    print("=" * 60)

    arr5 = [3, 1, 4]
    visualize_two_pointer(arr5.copy())

    print("\n" + "=" * 60)
    print("TEST 6: Edge Cases")
    print("=" * 60)

    edge_cases = [
        ([1, 2], False, "Too few elements"),
        ([1, 1, 2], True, "1 + 1 = 2"),
        ([0, 0, 0], True, "0 + 0 = 0"),
        ([5, 5, 10], True, "5 + 5 = 10"),
        ([], False, "Empty array"),
    ]

    for arr, expected, desc in edge_cases:
        result = findTriplet(arr.copy()) if arr else False
        status = "✅" if result == expected else "❌"
        print(status, desc.ljust(25), ":", arr, "→", result)

    print("\n" + "=" * 60)
    print("TEST 7: Different Approaches Comparison")
    print("=" * 60)

    arr7 = [1, 5, 3, 2, 8]
    print("Array:", arr7)
    print()

    result_sorted = findTriplet(arr7.copy())
    result_hash = findTriplet_hashset(arr7.copy())
    result_brute = findTriplet_brute_force(arr7.copy())

    print("Sort + Two Pointer:", result_sorted)
    print("HashSet:          ", result_hash)
    print("Brute Force:      ", result_brute)

    print("\n" + "=" * 60)
    print("TEST 8: Large Array")
    print("=" * 60)

    import random
    arr8 = [random.randint(1, 20) for _ in range(10)]
    print("Random array:", arr8)

    if findTriplet(arr8):
        print("Triplet exists!")
        triplets = findAllTriplets(arr8.copy())
        print("Examples:")
        for a, b, c in triplets[:3]:  # Show first 3
            print("  ", a, "+", b, "=", c)
    else:
        print("No triplet found")

    print("\n" + "=" * 60)
    print("TEST 9: Performance Test")
    print("=" * 60)

    import time

    test_arr = list(range(1, 101))

    # Two pointer
    start = time.time()
    findTriplet(test_arr.copy())
    time_two_pointer = time.time() - start

    # HashSet
    start = time.time()
    findTriplet_hashset(test_arr.copy())
    time_hash = time.time() - start

    # Brute force (on smaller array)
    small_arr = list(range(1, 21))
    start = time.time()
    findTriplet_brute_force(small_arr)
    time_brute = time.time() - start

    print("Array size: 100")
    print("Two Pointer: ", round(time_two_pointer, 6), "s")
    print("HashSet:     ", round(time_hash, 6), "s")
    print()
    print("Brute Force (size 20):", round(time_brute, 6), "s")

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Find if Sum of Two Elements Equals Third

Problem: arr[i] + arr[j] = arr[k]

Approach 1: Sort + Two Pointer (Optimal)
  1. Sort array
  2. For each element as target (from end)
  3. Use two pointers to find sum
  Time: O(n²), Space: O(1)

Approach 2: HashSet
  1. Store all elements in set
  2. Check all pairs
  3. See if sum exists
  Time: O(n²), Space: O(n)

Approach 3: Brute Force
  - Check all triplet combinations
  Time: O(n³), Space: O(1)

Best: Two Pointer approach (optimal time & space)
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print("""
Array: [5, 8, 3, 1]
Sorted: [1, 3, 5, 8]

Target = 8:
  [1, 3, 5, 8]
   L     R  T

  1 + 5 = 6 < 8, move L→

  [1, 3, 5, 8]
      L  R  T

  3 + 5 = 8 = 8 ✓ FOUND!

Answer: 3 + 5 = 8
    """)

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
1. Why sort?
   - Two pointer works on sorted array
   - Can efficiently find sum

2. Why target from end?
   - Larger values more likely to be sum
   - Optimization

3. Time complexity breakdown:
   - Sorting: O(n log n)
   - For each target: O(n)
   - Total targets: O(n)
   - Final: O(n²)

4. Similar problems:
   - Two Sum
   - Three Sum
   - Four Sum
   - 3Sum Closest
    """)

