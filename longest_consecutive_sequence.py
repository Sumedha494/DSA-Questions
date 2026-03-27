#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def longestConsecutive(nums):
    """
    Longest Consecutive Sequence

    Using HashSet - Optimal
    Time: O(n), Space: O(n)
    """
    if not nums:
        return 0

    num_set = set(nums)
    longest = 0

    for num in num_set:
        # Check if it's start of sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1

            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            longest = max(longest, current_length)

    return longest


def longestConsecutive_detailed(nums):
    """
    Step-by-step explanation
    """
    print("Longest Consecutive Sequence")
    print("=" * 60)
    print("Array:", nums)
    print()

    if not nums:
        return 0

    num_set = set(nums)
    print("Set (unique):", sorted(num_set))
    print()

    longest = 0

    print("Finding sequences:")
    print("-" * 60)

    for num in sorted(num_set):
        # Check if start of sequence
        if num - 1 not in num_set:
            print("\nStart of sequence:", num)

            current_num = num
            current_length = 1
            sequence = [num]

            # Build sequence
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
                sequence.append(current_num)

            print("  Sequence:", sequence)
            print("  Length:", current_length)

            if current_length > longest:
                longest = current_length
                print("  ✓ New longest!")

    print()
    print("=" * 60)
    print("Longest Length:", longest)

    return longest


def longestConsecutive_sorting(nums):
    """
    Using Sorting
    Time: O(n log n), Space: O(1)
    """
    if not nums:
        return 0

    nums.sort()
    longest = 1
    current_length = 1

    for i in range(1, len(nums)):
        # Skip duplicates
        if nums[i] == nums[i - 1]:
            continue

        # Check consecutive
        if nums[i] == nums[i - 1] + 1:
            current_length += 1
        else:
            longest = max(longest, current_length)
            current_length = 1

    return max(longest, current_length)


def longestConsecutive_with_sequence(nums):
    """
    Return length and the actual sequence
    """
    if not nums:
        return 0, []

    num_set = set(nums)
    longest = 0
    longest_seq = []

    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            sequence = [num]

            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
                sequence.append(current_num)

            if current_length > longest:
                longest = current_length
                longest_seq = sequence

    return longest, longest_seq


def findAllSequences(nums):
    """
    Find all consecutive sequences
    """
    if not nums:
        return []

    num_set = set(nums)
    sequences = []

    for num in sorted(num_set):
        if num - 1 not in num_set:
            current = num
            sequence = [num]

            while current + 1 in num_set:
                current += 1
                sequence.append(current)

            sequences.append(sequence)

    return sequences


def longestConsecutive_union_find(nums):
    """
    Using Union-Find (advanced)
    Time: O(n), Space: O(n)
    """
    if not nums:
        return 0

    parent = {}
    size = {}

    # Initialize
    for num in nums:
        if num not in parent:
            parent[num] = num
            size[num] = 1

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)

        if root_x != root_y:
            parent[root_x] = root_y
            size[root_y] += size[root_x]

    # Union consecutive numbers
    for num in nums:
        if num + 1 in parent:
            union(num, num + 1)

    return max(size.values())


def visualize_sequences(nums):
    """
    Visual representation
    """
    print("Consecutive Sequences Visualization")
    print("=" * 60)
    print("Array:", nums)
    print()

    sequences = findAllSequences(nums)

    print("All Consecutive Sequences:")
    print("-" * 40)

    for i, seq in enumerate(sequences, 1):
        print("Sequence", str(i) + ":", seq, "→ Length:", len(seq))

        # Visual bar
        bar = "─" * len(seq)
        print("           " + bar)

    print()

    if sequences:
        longest = max(sequences, key=len)
        print("Longest:", longest, "→ Length:", len(longest))

    return len(longest) if sequences else 0


def longestConsecutive_brute_force(nums):
    """
    Brute Force: For each number, count sequence
    Time: O(n³), Space: O(1)
    """
    if not nums:
        return 0

    longest = 0

    for num in nums:
        current_num = num
        current_length = 1

        # Count forward
        while current_num + 1 in nums:
            current_num += 1
            current_length += 1

        longest = max(longest, current_length)

    return longest


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ([100, 4, 200, 1, 3, 2], 4, "[1,2,3,4]"),
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9, "[0-8]"),
        ([9, 1, 4, 7, 3, 2, 8, 5, 6], 9, "[1-9]"),
        ([1, 2, 0, 1], 3, "[0,1,2]"),
    ]

    for nums, expected, desc in test_cases:
        result = longestConsecutive(nums)
        status = "✅" if result == expected else "❌"
        print(status, nums, "→", result, "-", desc)

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    nums2 = [100, 4, 200, 1, 3, 2]
    longestConsecutive_detailed(nums2)

    print("\n" + "=" * 60)
    print("TEST 3: With Sequence")
    print("=" * 60)

    nums3 = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
    length, sequence = longestConsecutive_with_sequence(nums3)

    print("Array:", nums3)
    print("Longest length:", length)
    print("Longest sequence:", sequence)

    print("\n" + "=" * 60)
    print("TEST 4: All Sequences")
    print("=" * 60)

    nums4 = [100, 4, 200, 1, 3, 2]
    print("Array:", nums4)

    all_seqs = findAllSequences(nums4)
    print("\nAll sequences:")
    for seq in all_seqs:
        print(" ", seq, "→ Length:", len(seq))

    print("\n" + "=" * 60)
    print("TEST 5: Visualization")
    print("=" * 60)

    nums5 = [1, 9, 3, 10, 4, 20, 2]
    visualize_sequences(nums5)

    print("\n" + "=" * 60)
    print("TEST 6: Edge Cases")
    print("=" * 60)

    edge_cases = [
        ([], 0, "Empty array"),
        ([1], 1, "Single element"),
        ([1, 1, 1], 1, "All duplicates"),
        ([1, 2, 3, 4, 5], 5, "Already consecutive"),
        ([5, 4, 3, 2, 1], 5, "Reverse order"),
        ([1, 3, 5, 7], 1, "No consecutive"),
        ([0, -1, 1], 3, "Negative numbers"),
    ]

    for nums, expected, desc in edge_cases:
        result = longestConsecutive(nums)
        status = "✅" if result == expected else "❌"
        print(status, desc.ljust(25), ":", result)

    print("\n" + "=" * 60)
    print("TEST 7: Sorting vs HashSet")
    print("=" * 60)

    import time
    import random

    test_arr = [random.randint(1, 1000) for _ in range(1000)]

    # HashSet method
    start = time.time()
    result_hash = longestConsecutive(test_arr)
    time_hash = time.time() - start

    # Sorting method
    start = time.time()
    result_sort = longestConsecutive_sorting(test_arr.copy())
    time_sort = time.time() - start

    print("Array size: 1000")
    print("HashSet (O(n)):    ", round(time_hash, 6), "s →", result_hash)
    print("Sorting (O(nlogn)):", round(time_sort, 6), "s →", result_sort)

    print("\n" + "=" * 60)
    print("TEST 8: Duplicates Handling")
    print("=" * 60)

    nums8 = [1, 2, 0, 1, 2, 3]
    print("Array:", nums8)
    print("With duplicates: [1, 2, 0, 1, 2, 3]")

    result = longestConsecutive(nums8)
    _, seq = longestConsecutive_with_sequence(nums8)

    print("Result:", result)
    print("Sequence:", seq, "(duplicates ignored)")

    print("\n" + "=" * 60)
    print("TEST 9: Negative Numbers")
    print("=" * 60)

    nums9 = [-1, -2, 0, 1, 2, -3]
    print("Array:", nums9)

    length, seq = longestConsecutive_with_sequence(nums9)
    print("Longest:", length)
    print("Sequence:", seq)

    print("\n" + "=" * 60)
    print("TEST 10: Large Gap")
    print("=" * 60)

    nums10 = [1, 2, 3, 100, 101, 102, 103, 104]
    print("Array:", nums10)

    all_seqs = findAllSequences(nums10)
    print("Sequences found:")
    for seq in all_seqs:
        print(" ", seq)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Longest Consecutive Sequence

Problem: Find length of longest consecutive elements

Approach 1: HashSet (Optimal) ⭐
  1. Put all numbers in set
  2. For each number n:
     - If n-1 NOT in set (start of sequence)
     - Count: n, n+1, n+2, ... while in set
  3. Track maximum length

  Time: O(n), Space: O(n)

  Why O(n)?
  - Each number visited at most twice
  - Once as start, once during counting

Approach 2: Sorting
  1. Sort array
  2. Count consecutive runs
  3. Skip duplicates

  Time: O(n log n), Space: O(1)

Approach 3: Union-Find
  - Union consecutive numbers
  - Track component sizes

  Time: O(n), Space: O(n)

Best: HashSet (O(n) time, simple)

LeetCode #128: Longest Consecutive Sequence
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print("""
Array: [100, 4, 200, 1, 3, 2]

Set: {1, 2, 3, 4, 100, 200}

Finding sequences:

  1: Is 0 in set? No → Start of sequence
     1 → 2 → 3 → 4 (length 4) ✓

  100: Is 99 in set? No → Start
       100 (length 1)

  200: Is 199 in set? No → Start
       200 (length 1)

Answer: 4 (sequence [1,2,3,4])
    """)

    print("\n" + "=" * 60)
    print("WHY O(n) TIME?")
    print("=" * 60)
    print("""
Key Insight: Each number checked AT MOST twice

Example: [1, 2, 3, 4]

Number 1:
  - Check if 0 exists (start check)
  - Count: 1→2→3→4 (sequence building)

Number 2:
  - Check if 1 exists → YES, skip!
  - Not counted again!

Number 3:
  - Check if 2 exists → YES, skip!

Number 4:
  - Check if 3 exists → YES, skip!

Total operations:
  - n checks for "is start?"
  - n increments during counting
  = 2n = O(n)
    """)

    print("\n" + "=" * 60)
    print("KEY PATTERNS")
    print("=" * 60)
    print("""
1. Use Set for O(1) lookup
2. Find START of sequence (num-1 not exists)
3. Count from start only
4. This prevents recounting!

Similar Problems:
  → Longest Consecutive Sequence
  → Number of Connected Components
  → Find Duplicate Number
  → Missing Number
    """)

