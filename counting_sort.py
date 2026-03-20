#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def countingSort(arr):
    """
    Basic Counting Sort
    Time: O(n + k), Space: O(k)
    """
    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    range_size = max_val - min_val + 1

    count = [0] * range_size

    # Count occurrences
    for num in arr:
        count[num - min_val] += 1

    # Build sorted array
    sorted_arr = []
    for i in range(range_size):
        sorted_arr.extend([i + min_val] * count[i])

    return sorted_arr


def countingSort_detailed(arr):
    """
    Step-by-step explanation
    """
    print("Counting Sort")
    print("=" * 50)
    print("Original:", arr)

    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    range_size = max_val - min_val + 1

    print("Min:", min_val, "Max:", max_val, "Range:", range_size)

    # Count
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1

    print("Count array:", count)

    # Build result
    result = []
    for i in range(range_size):
        val = i + min_val
        for j in range(count[i]):
            result.append(val)

    print("Sorted:", result)
    return result


def countingSort_stable(arr):
    """
    Stable version
    """
    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    range_size = max_val - min_val + 1

    count = [0] * range_size
    output = [0] * len(arr)



