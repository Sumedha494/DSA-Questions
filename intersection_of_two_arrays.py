#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def intersection(nums1, nums2):
    """
    Find intersection of two arrays (unique elements)

    Approach: Using Set
    Time Complexity: O(m + n)
    Space Complexity: O(min(m, n))
    """
    return list(set(nums1) & set(nums2))


def intersection_with_duplicates(nums1, nums2):
    """
    Find intersection with duplicates (LeetCode #350)
    Elements can appear multiple times

    Time Complexity: O(m + n)
    Space Complexity: O(min(m, n))
    """
    from collections import Counter

    count1 = Counter(nums1)
    count2 = Counter(nums2)

    result = []

    for num in count1:
        if num in count2:
            # Add minimum count times
            result.extend([num] * min(count1[num], count2[num]))

    return result


def intersection_sorted_arrays(nums1, nums2):
    """
    Intersection of sorted arrays using two pointers

    Time Complexity: O(m + n)
    Space Complexity: O(1) - excluding result
    """
    i, j = 0, 0
    result = []

    while i < len(nums1) and j < len(nums2):
        if nums1[i] < nums2[j]:
            i += 1
        elif nums1[i] > nums2[j]:
            j += 1
        else:
            # Add to result if not duplicate
            if not result or result[-1] != nums1[i]:
                result.append(nums1[i])
            i += 1
            j += 1

    return result


def intersection_brute_force(nums1, nums2):
    """
    Brute Force Approach
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    """
    result = []
    seen = set()

    for num in nums1:
        if num in nums2 and num not in seen:
            result.append(num)
            seen.add(num)

    return result


def intersection_binary_search(nums1, nums2):
    """
    Using Binary Search (when one array is sorted)

    Time Complexity: O(m log n) if nums2 is sorted
    Space Complexity: O(1) - excluding result
    """
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return True
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return False

    nums2_sorted = sorted(nums2)
    result = set()

    for num in nums1:
        if binary_search(nums2_sorted, num):
            result.add(num)

    return list(result)


def intersection_detailed(nums1, nums2):
    """
    Step-by-step explanation
    """
    print("Finding Intersection of Two Arrays")
    print("=" * 70)
    print("Array 1:", nums1)
    print("Array 2:", nums2)
    print()

    # Convert to sets
    print("Step 1: Convert to Sets")
    set1 = set(nums1)
    set2 = set(nums2)
    print("  Set 1:", set1)
    print("  Set 2:", set2)
    print()

    # Find intersection
    print("Step 2: Find Common Elements")
    intersection_set = set1 & set2
    print("  Intersection (set1 & set2):", intersection_set)
    print()

    # Convert to list
    result = list(intersection_set)
    print("Step 3: Convert to List")
    print("  Result:", result)
    print()

    print("=" * 70)
    print("Final Answer:", result)

    return result


def intersection_k_arrays(arrays):
    """
    Find intersection of k arrays

    Time Complexity: O(n * k) where n = avg array length
    Space Complexity: O(n)
    """
    if not arrays:
        return []

    # Start with first array
    result = set(arrays[0])

    # Intersect with remaining arrays
    for arr in arrays[1:]:
        result &= set(arr)

    return list(result)


def visualize_intersection(nums1, nums2):
    """
    Visual representation
    """
    print("Intersection Visualization")
    print("=" * 70)

    set1 = set(nums1)
    set2 = set(nums2)
    intersection_set = set1 & set2
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1

    print("Array 1:", nums1)
    print("Array 2:", nums2)
    print()

    print("Venn Diagram (Set representation):")
    print()
    print("  Only in nums1:", sorted(only_in_1))
    print("  Intersection: ", sorted(intersection_set), "← Answer")
    print("  Only in nums2:", sorted(only_in_2))
    print()

    print("Union (all unique):", sorted(set1 | set2))
    print()

    return list(intersection_set)


def intersection_with_indices(nums1, nums2):
    """
    Return intersection with indices from both arrays
    """
    result = {}
    set2 = set(nums2)

    for i, num in enumerate(nums1):
        if num in set2 and num not in result:
            # Find first occurrence in nums2
            j = nums2.index(num)
            result[num] = (i, j)

    return result


def intersection_multiple_approaches(nums1, nums2):
    """
    Compare all approaches
    """
    print("Comparing Different Approaches")
    print("=" * 70)

    approaches = [
        ("Set Operation", lambda: list(set(nums1) & set(nums2))),
        ("Two Pointers", lambda: intersection_sorted_arrays(sorted(nums1), sorted(nums2))),
        ("Brute Force", lambda: intersection_brute_force(nums1, nums2)),
        ("Binary Search", lambda: intersection_binary_search(nums1, nums2)),
    ]

    for name, func in approaches:
        import time
        start = time.time()
        result = func()
        elapsed = time.time() - start

        print(name.ljust(20), ":", sorted(result), 
              "(" + str(round(elapsed, 8)) + "s)")


def intersection_count_frequency(nums1, nums2):
    """
    Return intersection with frequency count
    """
    from collections import Counter

    count1 = Counter(nums1)
    count2 = Counter(nums2)

    result = {}

    for num in count1:
        if num in count2:
            result[num] = {
                'in_nums1': count1[num],
                'in_nums2': count2[num],
                'in_intersection': min(count1[num], count2[num])
            }

    return result


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEST 1: Basic Intersection (Unique)")
    print("=" * 70)

    nums1 = [1, 2, 2, 1]
    nums2 = [2, 2]
    print("nums1:", nums1)
    print("nums2:", nums2)
    result = intersection(nums1, nums2)
    print("Intersection:", result)
    print("Expected: [2]")

    print("\n" + "=" * 70)
    print("TEST 2: Intersection with Duplicates")
    print("=" * 70)

    nums1 = [4, 9, 5]
    nums2 = [9, 4, 9, 8, 4]
    print("nums1:", nums1)
    print("nums2:", nums2)
    result1 = intersection(nums1, nums2)
    result2 = intersection_with_duplicates(nums1, nums2)
    print("Unique intersection:    ", sorted(result1))
    print("With duplicates:        ", sorted(result2))

    print("\n" + "=" * 70)
    print("TEST 3: Detailed Step-by-Step")
    print("=" * 70)

    nums1 = [1, 2, 2, 1]
    nums2 = [2, 2]
    intersection_detailed(nums1, nums2)

    print("\n" + "=" * 70)
    print("TEST 4: Sorted Arrays (Two Pointers)")
    print("=" * 70)

    nums1 = [1, 2, 3, 4, 5]
    nums2 = [3, 4, 5, 6, 7]
    print("nums1 (sorted):", nums1)
    print("nums2 (sorted):", nums2)
    result = intersection_sorted_arrays(nums1, nums2)
    print("Intersection:  ", result)
    print("Expected: [3, 4, 5]")

    print("\n" + "=" * 70)
    print("TEST 5: Visualization")
    print("=" * 70)

    nums1 = [1, 2, 3, 4, 5]
    nums2 = [4, 5, 6, 7, 8]
    visualize_intersection(nums1, nums2)

    print("\n" + "=" * 70)
    print("TEST 6: Edge Cases")
    print("=" * 70)

    edge_cases = [
        ([1], [1], [1], "Single element match"),
        ([1], [2], [], "Single element no match"),
        ([], [1, 2], [], "Empty first array"),
        ([1, 2], [], [], "Empty second array"),
        ([], [], [], "Both empty"),
        ([1, 2, 3], [1, 2, 3], [1, 2, 3], "Identical arrays"),
        ([1, 2, 3], [4, 5, 6], [], "No common elements"),
        ([1, 1, 1], [1], [1], "All duplicates"),
    ]

    for nums1, nums2, expected, desc in edge_cases:
        result = sorted(intersection(nums1, nums2))
        expected_sorted = sorted(expected)
        status = "✅" if result == expected_sorted else "❌"
        print(status, desc.ljust(30), ":", nums1, "∩", nums2, "→", result)

    print("\n" + "=" * 70)
    print("TEST 7: K Arrays Intersection")
    print("=" * 70)

    arrays = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8]
    ]

    print("Arrays:")
    for i, arr in enumerate(arrays):
        print("  Array", i + 1, ":", arr)

    result = intersection_k_arrays(arrays)
    print("\nIntersection of all:", sorted(result))
    print("Expected: [4, 5]")

    print("\n" + "=" * 70)
    print("TEST 8: With Indices")
    print("=" * 70)

    nums1 = [4, 9, 5]
    nums2 = [9, 4, 9, 8, 4]
    print("nums1:", nums1)
    print("nums2:", nums2)
    result = intersection_with_indices(nums1, nums2)
    print("\nIntersection with indices:")
    for num, (i, j) in result.items():
        print("  ", num, ": nums1[" + str(i) + "], nums2[" + str(j) + "]")

    print("\n" + "=" * 70)
    print("TEST 9: Frequency Count")
    print("=" * 70)

    nums1 = [1, 2, 2, 1, 3]
    nums2 = [2, 2, 3, 3]
    print("nums1:", nums1)
    print("nums2:", nums2)
    result = intersection_count_frequency(nums1, nums2)
    print("\nFrequency Analysis:")
    for num, counts in sorted(result.items()):
        print("  ", num, ":", counts)

    print("\n" + "=" * 70)
    print("TEST 10: Performance Comparison")
    print("=" * 70)

    import random
    large_nums1 = [random.randint(1, 100) for _ in range(1000)]
    large_nums2 = [random.randint(1, 100) for _ in range(1000)]

    print("Array sizes: 1000 each")
    print()
    intersection_multiple_approaches(large_nums1, large_nums2)

    print("\n" + "=" * 70)
    print("TEST 11: LeetCode Style Problems")
    print("=" * 70)

    # LeetCode #349: Intersection of Two Arrays
    print("Problem 1: Intersection (LeetCode #349)")
    nums1 = [1, 2, 2, 1]
    nums2 = [2, 2]
    print("  nums1:", nums1)
    print("  nums2:", nums2)
    print("  Output:", intersection(nums1, nums2))

    # LeetCode #350: Intersection of Two Arrays II
    print("\nProblem 2: Intersection II (LeetCode #350)")
    nums1 = [1, 2, 2, 1]
    nums2 = [2, 2]
    print("  nums1:", nums1)
    print("  nums2:", nums2)
    print("  Output:", intersection_with_duplicates(nums1, nums2))

    print("\n" + "=" * 70)
    print("TEST 12: Set Operations")
    print("=" * 70)

    nums1 = [1, 2, 3, 4, 5]
    nums2 = [4, 5, 6, 7, 8]

    set1 = set(nums1)
    set2 = set(nums2)

    print("nums1:", nums1)
    print("nums2:", nums2)
    print()
    print("Union (|):         ", sorted(set1 | set2))
    print("Intersection (&):  ", sorted(set1 & set2))
    print("Difference (-):    ", sorted(set1 - set2))
    print("Symmetric Diff (^):", sorted(set1 ^ set2))

    print("\n" + "=" * 70)
    print("ALGORITHM SUMMARY")
    print("=" * 70)
    print("""
Intersection of Two Arrays - Multiple Approaches

Problem: Find common elements between two arrays

Approaches:

1. Set Intersection (Best for unique elements):
   - Convert both to sets
   - Use set intersection (&)
   - Time: O(m + n), Space: O(m + n)
   - ⭐ Most pythonic and efficient

2. Two Pointers (For sorted arrays):
   - Sort both arrays
   - Use two pointers
   - Time: O(m log m + n log n), Space: O(1)
   - Good when arrays already sorted

3. HashMap/Counter (For duplicates):
   - Count frequencies
   - Take minimum counts
   - Time: O(m + n), Space: O(min(m, n))
   - ⭐ Best when duplicates matter

4. Binary Search:
   - Sort one array
   - Binary search for each element
   - Time: O(m log n), Space: O(1)
   - Good when one array much smaller

5. Brute Force:
   - Check each element in other array
   - Time: O(m * n), Space: O(1)
   - Simple but slow

Variations:
✓ Unique elements (LeetCode #349)
✓ With duplicates (LeetCode #350)
✓ K arrays intersection
✓ With indices/frequencies

Time Complexity: O(m + n) with set/hashmap ⚡
Space Complexity: O(min(m, n))

Key Points:
- Set operation is fastest for unique
- Counter/HashMap for duplicates
- Two pointers for sorted arrays
- Consider memory constraints
    """)

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Intersection vs Union:
   Intersection: Common elements (∩)
   Union: All elements (∪)

2. Set Operations in Python:
   & → intersection
   | → union
   - → difference
   ^ → symmetric difference

3. When to use what:
   - Unique elements? → set(&)
   - Duplicates matter? → Counter
   - Arrays sorted? → Two pointers
   - Space critical? → Binary search

4. Follow-up questions:
   Q: What if arrays are sorted?
   A: Use two pointers (no extra space)

   Q: What if nums1 size << nums2 size?
   A: Binary search nums1 elements in nums2

   Q: What if duplicates matter?
   A: Use Counter, take min frequency

5. Real-world applications:
   - Common friends on social media
   - Common products in inventory
   - Shared interests/tags
   - Database JOIN operations
    """)

