#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def countSubarraysWithSum(arr, target):
    """
    Count subarrays with given sum

    Using HashMap (Prefix Sum)
    Time: O(n), Space: O(n)
    """
    if not arr:
        return 0

    count = 0
    prefix_sum = 0
    sum_count = {0: 1}  # Handle subarrays starting from index 0

    for num in arr:
        prefix_sum += num

        # If (prefix_sum - target) exists, those many subarrays end here
        if prefix_sum - target in sum_count:
            count += sum_count[prefix_sum - target]

        # Store current prefix sum
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1

    return count


def countSubarraysWithSum_detailed(arr, target):
    """
    Step-by-step explanation
    """
    print("Count Subarrays with Given Sum")
    print("=" * 60)
    print("Array:", arr)
    print("Target Sum:", target)
    print()

    if not arr:
        return 0

    count = 0
    prefix_sum = 0
    sum_count = {0: 1}

    print("Using Prefix Sum and HashMap:")
    print("-" * 60)
    print("Step | Element | Prefix | Looking for | Found | Count")
    print("-" * 60)

    for i, num in enumerate(arr):
        prefix_sum += num

        # What we're looking for
        needed = prefix_sum - target

        # How many subarrays end here
        found = sum_count.get(needed, 0)
        count += found

        print(str(i).rjust(4), "|", 
              str(num).rjust(7), "|", 
              str(prefix_sum).rjust(6), "|", 
              str(needed).rjust(11), "|", 
              str(found).rjust(5), "|", 
              str(count).rjust(5))

        # Update hashmap
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1

    print("-" * 60)
    print("\nHashMap state:", sum_count)
    print()
    print("=" * 60)
    print("Total subarrays:", count)

    return count


def countSubarraysWithSum_bruteforce(arr, target):
    """
    Brute Force: Check all subarrays
    Time: O(n²), Space: O(1)
    """
    if not arr:
        return 0

    n = len(arr)
    count = 0

    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += arr[j]

            if current_sum == target:
                count += 1

    return count


def findAllSubarraysWithSum(arr, target):
    """
    Find all actual subarrays (not just count)
    """
    if not arr:
        return []

    n = len(arr)
    result = []

    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += arr[j]

            if current_sum == target:
                result.append(arr[i:j+1])

    return result


def countSubarraysWithSum_with_indices(arr, target):
    """
    Return count and all subarray indices
    """
    if not arr:
        return 0, []

    prefix_sum = 0
    sum_indices = {0: [-1]}  # -1 means start from index 0
    result = []

    for i, num in enumerate(arr):
        prefix_sum += num

        needed = prefix_sum - target

        if needed in sum_indices:
            for start_idx in sum_indices[needed]:
                result.append((start_idx + 1, i))

        if prefix_sum not in sum_indices:
            sum_indices[prefix_sum] = []
        sum_indices[prefix_sum].append(i)

    return len(result), result


def visualizeSubarrays(arr, target):
    """
    Visual representation
    """
    print("Subarray Visualization")
    print("=" * 60)
    print("Array:", arr)
    print("Target:", target)
    print()

    subarrays = findAllSubarraysWithSum(arr, target)

    if not subarrays:
        print("No subarrays found!")
        return 0

    print("All subarrays with sum =", target, ":")
    print("-" * 40)

    for i, sub in enumerate(subarrays, 1):
        total = sum(sub)
        bar = "█" * len(sub)
        print(str(i).rjust(2), ".", sub, "=", total, bar)

    print()
    print("Total count:", len(subarrays))

    return len(subarrays)


def countSubarraysWithSum_prefix_visual(arr, target):
    """
    Visual explanation of prefix sum method
    """
    print("Prefix Sum Method Visualization")
    print("=" * 60)
    print("Array:", arr)
    print("Target:", target)
    print()

    prefix = [0]
    for num in arr:
        prefix.append(prefix[-1] + num)

    print("Prefix sums:", prefix)
    print()

    print("Why this works:")
    print("-" * 40)
    print("If prefix[j] - prefix[i] = target")
    print("Then subarray arr[i:j] has sum = target")
    print()

    count = 0
    sum_count = {0: 1}

    print("Finding pairs:")
    for i in range(1, len(prefix)):
        current = prefix[i]
        needed = current - target

        if needed in sum_count:
            print("prefix[" + str(i) + "] =", current, 
                  ", need", needed, "→ Found", sum_count[needed], "subarray(s)")
            count += sum_count[needed]

        sum_count[current] = sum_count.get(current, 0) + 1

    print()
    print("Total:", count)

    return count


def countSubarraysInRange(arr, min_sum, max_sum):
    """
    Count subarrays with sum in range [min_sum, max_sum]
    """
    if not arr:
        return 0

    count = 0
    n = len(arr)

    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += arr[j]

            if min_sum <= current_sum <= max_sum:
                count += 1

    return count


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ([1, 1, 1], 2, 2),
        ([1, 2, 3], 3, 2),
        ([1, -1, 0], 0, 3),
        ([3, 4, 7, 2, -3, 1, 4, 2], 7, 4),
    ]

    for arr, target, expected in test_cases:
        result = countSubarraysWithSum(arr, target)
        status = "✅" if result == expected else "❌"
        print(status, "arr:", arr, ", target:", target, "→", result)

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    countSubarraysWithSum_detailed([1, 1, 1], 2)

    print("\n" + "=" * 60)
    print("TEST 3: Find Actual Subarrays")
    print("=" * 60)

    arr3 = [1, 2, 3]
    target3 = 3

    subarrays = findAllSubarraysWithSum(arr3, target3)
    print("Array:", arr3)
    print("Target:", target3)
    print("Subarrays:")
    for sub in subarrays:
        print(" ", sub, "=", sum(sub))

    print("\n" + "=" * 60)
    print("TEST 4: With Indices")
    print("=" * 60)

    arr4 = [1, 1, 1]
    target4 = 2
    count4, indices4 = countSubarraysWithSum_with_indices(arr4, target4)

    print("Array:", arr4)
    print("Target:", target4)
    print("Count:", count4)
    print("Indices (start, end):", indices4)

    for start, end in indices4:
        print("  arr[" + str(start) + ":" + str(end+1) + "] =", 
              arr4[start:end+1])

    print("\n" + "=" * 60)
    print("TEST 5: Visualization")
    print("=" * 60)

    visualizeSubarrays([3, 4, 7, 2, -3, 1, 4, 2], 7)

    print("\n" + "=" * 60)
    print("TEST 6: Prefix Sum Visual")
    print("=" * 60)

    countSubarraysWithSum_prefix_visual([1, 2, 3], 3)

    print("\n" + "=" * 60)
    print("TEST 7: Edge Cases")
    print("=" * 60)

    # Empty array
    print("Empty array:", countSubarraysWithSum([], 0))

    # Single element match
    print("Single element [5], target=5:", 
          countSubarraysWithSum([5], 5))

    # Single element no match
    print("Single element [5], target=3:", 
          countSubarraysWithSum([5], 3))

    # All zeros
    print("All zeros [0,0,0], target=0:", 
          countSubarraysWithSum([0, 0, 0], 0))

    # No solution
    print("No solution [1,2,3], target=10:", 
          countSubarraysWithSum([1, 2, 3], 10))

    print("\n" + "=" * 60)
    print("TEST 8: Negative Numbers")
    print("=" * 60)

    arr8 = [1, -1, 0]
    target8 = 0

    print("Array:", arr8)
    print("Target:", target8)
    count8 = countSubarraysWithSum(arr8, target8)
    subs8 = findAllSubarraysWithSum(arr8, target8)

    print("Count:", count8)
    print("Subarrays:")
    for sub in subs8:
        print(" ", sub)

    print("\n" + "=" * 60)
    print("TEST 9: Compare Methods")
    print("=" * 60)

    import time
    import random

    test_arr = [random.randint(-5, 5) for _ in range(1000)]
    target = 5

    # HashMap method
    start = time.time()
    result_hash = countSubarraysWithSum(test_arr, target)
    time_hash = time.time() - start

    # Brute force (on smaller array)
    small_arr = test_arr[:100]
    start = time.time()
    result_brute = countSubarraysWithSum_bruteforce(small_arr, target)
    time_brute = time.time() - start

    print("Array size: 1000")
    print("HashMap (O(n)):    ", round(time_hash, 6), "s →", result_hash)
    print()
    print("Array size: 100")
    print("Brute Force (O(n²)):", round(time_brute, 6), "s →", result_brute)

    print("\n" + "=" * 60)
    print("TEST 10: Sum in Range")
    print("=" * 60)

    arr10 = [1, 2, 3, 4, 5]
    min_sum = 3
    max_sum = 7

    print("Array:", arr10)
    print("Range: [" + str(min_sum) + ", " + str(max_sum) + "]")

    count10 = countSubarraysInRange(arr10, min_sum, max_sum)
    print("Count:", count10)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Count Subarrays with Given Sum

Problem: Count subarrays with sum = target

Method 1: HashMap + Prefix Sum (Optimal) ⭐
  Key Idea:
    If prefix_sum[j] - prefix_sum[i] = target
    Then subarray arr[i+1:j+1] has sum = target

  Algorithm:
    1. Track prefix sums in hashmap
    2. For each position:
       - If (current_sum - target) exists in map
       - That many subarrays end here
    3. Add current sum to hashmap

  Time: O(n), Space: O(n)

Method 2: Brute Force
  - Check all subarrays
  Time: O(n²), Space: O(1)

Similar Problems:
  → Subarray Sum Equals K (LeetCode #560)
  → Continuous Subarray Sum
  → Binary Subarrays With Sum

LeetCode #560: Subarray Sum Equals K
    """)

    print("\n" + "=" * 60)
    print("WHY PREFIX SUM WORKS")
    print("=" * 60)
    print("""
Array: [1, 2, 3], Target: 3

Prefix sums: [0, 1, 3, 6]
             0  1  2  3  (indices)

Finding subarrays with sum = 3:

At index 2 (prefix = 3):
  Need: 3 - 3 = 0
  Found: prefix[0] = 0
  Subarray: arr[0:2] = [1, 2] ✓

At index 3 (prefix = 6):
  Need: 6 - 3 = 3
  Found: prefix[2] = 3
  Subarray: arr[2:3] = [3] ✓

Total: 2 subarrays

Formula:
  If prefix[j] - prefix[i] = target
  Then arr[i:j] has sum = target
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print("""
Array: [1, 1, 1], Target: 2

Step-by-step:
HashMap: {0: 1}  (initially)

i=0, num=1, prefix=1
  Looking for: 1-2 = -1 (not found)
  Count = 0
  HashMap: {0: 1, 1: 1}

i=1, num=1, prefix=2
  Looking for: 2-2 = 0 (found 1 time!)
  Count = 1  [subarray: [1,1]]
  HashMap: {0: 1, 1: 1, 2: 1}

i=2, num=1, prefix=3
  Looking for: 3-2 = 1 (found 1 time!)
  Count = 2  [subarray: [1,1]]
  HashMap: {0: 1, 1: 1, 2: 1, 3: 1}

Total: 2 subarrays
  [1, 1] at indices [0,1]
  [1, 1] at indices [1,2]
    """)

