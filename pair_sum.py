#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def two_sum_hashmap(nums, target):
    """
    Time: O(n) - Single pass
    Space: O(n) - Hash map storage
    """
    seen = {}  # {value: index}

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]

        seen[num] = i

    return []  # No solution found

# --- Example ---
nums = [2, 7, 11, 15]
target = 9
print(f"Indices: {two_sum_hashmap(nums, target)}")  # [0, 1]


# In[ ]:


def two_sum_sorted(nums, target):
    """
    Time: O(n log n) - Due to sorting
    Space: O(1) - No extra space (if sorting in-place)
    """
    # Create list of (value, original_index) to track indices after sorting
    indexed_nums = [(val, idx) for idx, val in enumerate(nums)]
    indexed_nums.sort()  # Sort by value

    left, right = 0, len(indexed_nums) - 1

    while left < right:
        current_sum = indexed_nums[left][0] + indexed_nums[right][0]

        if current_sum == target:
            return [indexed_nums[left][1], indexed_nums[right][1]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []

# --- Example ---
nums = [3, 2, 4]
target = 6
print(f"Indices (Sorted): {two_sum_sorted(nums, target)}")  # [1, 2]


# In[ ]:


def two_sum_brute(nums, target):
    """
    Time: O(n^2) - Nested loops
    Space: O(1)
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


# In[ ]:


def all_pairs_sum(nums, target):
    """
    Returns a list of all unique pairs (values, not indices)
    """
    nums.sort()
    left, right = 0, len(nums) - 1
    pairs = []

    while left < right:
        current_sum = nums[left] + nums[right]

        if current_sum == target:
            pairs.append((nums[left], nums[right]))
            left += 1
            right -= 1
            # Skip duplicates
            while left < right and nums[left] == nums[left - 1]:
                left += 1
            while left < right and nums[right] == nums[right + 1]:
                right -= 1

        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return pairs

# --- Example ---
nums = [1, 2, 3, 4, 5, 6, 7]
target = 8
print(f"All Pairs: {all_pairs_sum(nums, target)}")
# Output: [(1, 7), (2, 6), (3, 5)]

