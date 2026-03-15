#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def dutchNationalFlag(arr):
    """
    Dutch National Flag Algorithm (3-way partitioning)
    Sort array of 0s, 1s, and 2s

    Time Complexity: O(n)
    Space Complexity: O(1)

    Three pointers approach:
    - low: boundary of 0s
    - mid: current element
    - high: boundary of 2s
    """
    low = 0
    mid = 0
    high = len(arr) - 1

    while mid <= high:
        if arr[mid] == 0:
            # Swap with low and move both pointers
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1

        elif arr[mid] == 1:
            # 1 is in correct region, just move mid
            mid += 1

        else:  # arr[mid] == 2
            # Swap with high and move high only
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

    return arr


def dutchNationalFlag_detailed(arr):
    """
    Step-by-step explanation ke saath
    """
    print("Dutch National Flag Algorithm")
    print("=" * 70)
    print("Original Array:", arr)
    print("\nGoal: [All 0s | All 1s | All 2s]")
    print("=" * 70)

    low = 0
    mid = 0
    high = len(arr) - 1
    step = 1

    print("\nInitial State:")
    print("low =", low, ", mid =", mid, ", high =", high)
    print()

    while mid <= high:
        print("Step", step, ":")
        print("Array:", arr)

        # Visual representation
        visual = []
        for i in range(len(arr)):
            if i == low and i == mid:
                visual.append("[L,M:" + str(arr[i]) + "]")
            elif i == low:
                visual.append("[L:" + str(arr[i]) + "]")
            elif i == mid:
                visual.append("[M:" + str(arr[i]) + "]")
            elif i == high:
                visual.append("[H:" + str(arr[i]) + "]")
            else:
                visual.append(" " + str(arr[i]) + " ")

        print("Visual:", " ".join(visual))
        print("Pointers: low=" + str(low) + ", mid=" + str(mid) + ", high=" + str(high))

        if arr[mid] == 0:
            print("→ Found 0 at mid, swap with low")
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
            print("  Moved: low++ and mid++")

        elif arr[mid] == 1:
            print("→ Found 1 at mid, it's in correct region")
            mid += 1
            print("  Moved: mid++")

        else:  # arr[mid] == 2
            print("→ Found 2 at mid, swap with high")
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
            print("  Moved: high--")

        print()
        step += 1

    print("=" * 70)
    print("Final Sorted Array:", arr)
    print("\nRegions:")
    print("0s: arr[0:" + str(low) + "]")
    print("1s: arr[" + str(low) + ":" + str(mid) + "]")
    print("2s: arr[" + str(mid) + ":" + str(len(arr)) + "]")

    return arr


def sort_colors(nums):
    """
    LeetCode #75 style - Sort Colors
    Same as Dutch National Flag
    """
    low = 0
    mid = 0
    high = len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1

    return nums


def partition_around_pivot(arr, pivot):
    """
    Generalized version: Partition around any pivot value
    Elements: [< pivot | = pivot | > pivot]
    """
    low = 0
    mid = 0
    high = len(arr) - 1

    while mid <= high:
        if arr[mid] < pivot:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == pivot:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

    return arr


def count_012(arr):
    """
    Count approach (not in-place, uses extra space)
    """
    count_0 = arr.count(0)
    count_1 = arr.count(1)
    count_2 = arr.count(2)

    result = [0] * count_0 + [1] * count_1 + [2] * count_2
    return result


def visualize_sorting_process(arr):
    """
    Visual representation of sorting process
    """
    print("Dutch National Flag - Visual Sorting")
    print("=" * 70)
    print("Initial:", arr)
    print()

    low = 0
    mid = 0
    high = len(arr) - 1

    # Print regions
    def print_regions():
        print("Regions:")
        print("  0s region: [0 to", low - 1, "]", "→", arr[:low] if low > 0 else "[]")
        print("  Unknown:   [" + str(low) + " to", high, "]", "→", arr[low:high + 1] if low <= high else "[]")
        print("  2s region: [" + str(high + 1) + " to", len(arr) - 1, "]", "→", arr[high + 1:] if high < len(arr) - 1 else "[]")
        print()

    step = 0
    print_regions()

    while mid <= high:
        step += 1
        print("Step", step, "- arr[mid] =", arr[mid])

        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            print("  Action: Swap arr[" + str(low) + "] with arr[" + str(mid) + "]")
            low += 1
            mid += 1
        elif arr[mid] == 1:
            print("  Action: Just move mid pointer")
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            print("  Action: Swap arr[" + str(mid) + "] with arr[" + str(high) + "]")
            high -= 1

        print("  Array:", arr)
        print_regions()

    print("=" * 70)
    print("Final Sorted:", arr)


def dutch_flag_with_stats(arr):
    """
    Track swaps and comparisons
    """
    low = 0
    mid = 0
    high = len(arr) - 1
    swaps = 0
    comparisons = 0

    while mid <= high:
        comparisons += 1

        if arr[mid] == 0:
            if low != mid:
                arr[low], arr[mid] = arr[mid], arr[low]
                swaps += 1
            low += 1
            mid += 1

        elif arr[mid] == 1:
            mid += 1

        else:
            if mid != high:
                arr[mid], arr[high] = arr[high], arr[mid]
                swaps += 1
            high -= 1

    return arr, comparisons, swaps


def three_way_quicksort_partition(arr, pivot_index):
    """
    3-way partitioning for QuickSort optimization
    Useful when array has many duplicates
    """
    pivot = arr[pivot_index]
    low = 0
    mid = 0
    high = len(arr) - 1

    while mid <= high:
        if arr[mid] < pivot:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == pivot:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

    return arr, low, high


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEST 1: Basic Dutch National Flag")
    print("=" * 70)
    arr1 = [2, 0, 2, 1, 1, 0]
    print("Original:", arr1)
    result1 = dutchNationalFlag(arr1.copy())
    print("Sorted:  ", result1)
    print("Expected: [0, 0, 1, 1, 2, 2]")

    print("\n" + "=" * 70)
    print("TEST 2: Detailed Step-by-Step")
    print("=" * 70)
    arr2 = [1, 2, 0, 2, 1, 0]
    dutchNationalFlag_detailed(arr2.copy())

    print("\n" + "=" * 70)
    print("TEST 3: LeetCode #75 - Sort Colors")
    print("=" * 70)
    arr3 = [2, 0, 2, 1, 1, 0]
    print("Original:", arr3)
    result3 = sort_colors(arr3.copy())
    print("Sorted:  ", result3)

    print("\n" + "=" * 70)
    print("TEST 4: Partition Around Pivot")
    print("=" * 70)
    arr4 = [3, 5, 2, 5, 1, 5, 7, 5, 9]
    pivot = 5
    print("Original:", arr4)
    print("Pivot:   ", pivot)
    result4 = partition_around_pivot(arr4.copy(), pivot)
    print("Result:  ", result4)
    print("Format: [< 5 | = 5 | > 5]")

    print("\n" + "=" * 70)
    print("TEST 5: Visualization")
    print("=" * 70)
    arr5 = [2, 0, 1, 2, 1, 0]
    visualize_sorting_process(arr5.copy())

    print("\n" + "=" * 70)
    print("TEST 6: With Statistics")
    print("=" * 70)
    arr6 = [2, 0, 2, 1, 1, 0, 2, 1, 0]
    original = arr6.copy()
    result6, comps, swaps = dutch_flag_with_stats(arr6.copy())
    print("Original:   ", original)
    print("Sorted:     ", result6)
    print("Comparisons:", comps)
    print("Swaps:      ", swaps)

    print("\n" + "=" * 70)
    print("TEST 7: Edge Cases")
    print("=" * 70)

    edge_cases = [
        ([0], "Single 0"),
        ([1], "Single 1"),
        ([2], "Single 2"),
        ([0, 0, 0], "All 0s"),
        ([1, 1, 1], "All 1s"),
        ([2, 2, 2], "All 2s"),
        ([0, 1, 2], "Already sorted"),
        ([2, 1, 0], "Reverse sorted"),
        ([1, 0, 1, 0, 1, 0], "Only 0s and 1s"),
        ([2, 1, 2, 1, 2, 1], "Only 1s and 2s"),
        ([], "Empty array"),
    ]

    for arr, desc in edge_cases:
        original = arr.copy()
        result = dutchNationalFlag(arr.copy()) if arr else arr
        print(desc.ljust(20), ":", original, "→", result)

    print("\n" + "=" * 70)
    print("TEST 8: Comparison with Count Method")
    print("=" * 70)
    import time
    import random

    # Large array
    large_arr = [random.randint(0, 2) for _ in range(10000)]

    # Dutch National Flag
    arr_dnf = large_arr.copy()
    start = time.time()
    dutchNationalFlag(arr_dnf)
    time_dnf = time.time() - start

    # Count method
    arr_count = large_arr.copy()
    start = time.time()
    result_count = count_012(arr_count)
    time_count = time.time() - start

    print("Array size: 10,000 elements")
    print("Dutch National Flag (in-place):", round(time_dnf, 6), "seconds")
    print("Count Method (extra space):    ", round(time_count, 6), "seconds")
    print("\nBoth are O(n) but DNF is in-place!")

    print("\n" + "=" * 70)
    print("TEST 9: Different Scenarios")
    print("=" * 70)

    scenarios = [
        ([2, 2, 2, 0, 0, 0, 1, 1, 1], "Grouped by value"),
        ([0, 1, 2, 0, 1, 2, 0, 1, 2], "Repeating pattern"),
        ([1, 1, 1, 1, 0, 2], "Mostly 1s"),
        ([0, 2, 0, 2, 0, 2], "No 1s"),
    ]

    for arr, desc in scenarios:
        original = arr.copy()
        result = dutchNationalFlag(arr.copy())
        print(desc)
        print("  Original:", original)
        print("  Sorted:  ", result)
        print()

    print("=" * 70)
    print("TEST 10: 3-Way QuickSort Partition")
    print("=" * 70)
    arr10 = [4, 9, 4, 4, 1, 9, 4, 4, 9, 4, 4, 1, 4]
    print("Original:", arr10)
    print("Pivot: arr[0] =", arr10[0])
    result10, low, high = three_way_quicksort_partition(arr10.copy(), 0)
    print("Result:  ", result10)
    print("Partition indices: low=" + str(low) + ", high=" + str(high))
    print("Less than pivot:", result10[:low])
    print("Equal to pivot: ", result10[low:high + 1])
    print("Greater than pivot:", result10[high + 1:])

    print("\n" + "=" * 70)
    print("ALGORITHM SUMMARY")
    print("=" * 70)
    print("""
Dutch National Flag Algorithm (Edsger Dijkstra)

Purpose: Sort array of 0s, 1s, and 2s in O(n) time, O(1) space

Key Concept: Three-way partitioning using 3 pointers
- low:  Everything before this is 0
- mid:  Current element being examined
- high: Everything after this is 2

Rules:
1. arr[mid] == 0: Swap with low, move both low++ and mid++
2. arr[mid] == 1: Just move mid++
3. arr[mid] == 2: Swap with high, move only high--

Time Complexity: O(n) - Single pass
Space Complexity: O(1) - In-place

Applications:
✓ Sort colors (LeetCode #75)
✓ 3-way partitioning in QuickSort
✓ Grouping elements into categories
✓ Segregation problems
    """)

