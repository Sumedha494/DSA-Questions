#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def selectionSort(arr):
    """
    Basic Selection Sort
    Time Complexity: O(n²) - All cases
    Space Complexity: O(1)
    """
    n = len(arr)

    for i in range(n):
        # Minimum element ka index dhundho
        min_idx = i

        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # Minimum element ko current position pe swap karo
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr


def selectionSort_detailed(arr):
    """
    Step-by-step explanation ke saath
    """
    print("Original Array:", arr)
    print("=" * 60)

    n = len(arr)

    for i in range(n):
        min_idx = i
        print("\nPass", i + 1, ":")
        print("Current state:", arr)
        print("Finding minimum in unsorted part:", arr[i:])

        # Minimum dhundho
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        print("Minimum found:", arr[min_idx], "at index", min_idx)

        # Swap karo
        if min_idx != i:
            print("Swapping", arr[i], "with", arr[min_idx])
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        else:
            print("No swap needed")

        print("After this pass:", arr)
        print("Sorted portion:", arr[:i+1])

    print("\n" + "=" * 60)
    print("Final Sorted Array:", arr)
    return arr


def selectionSort_descending(arr):
    """
    Descending order mein sort
    """
    n = len(arr)

    for i in range(n):
        max_idx = i

        for j in range(i + 1, n):
            if arr[j] > arr[max_idx]:
                max_idx = j

        arr[i], arr[max_idx] = arr[max_idx], arr[i]

    return arr


def selectionSort_with_stats(arr):
    """
    Comparisons aur Swaps count karo
    """
    n = len(arr)
    comparisons = 0
    swaps = 0

    for i in range(n):
        min_idx = i

        for j in range(i + 1, n):
            comparisons += 1
            if arr[j] < arr[min_idx]:
                min_idx = j

        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            swaps += 1

    return arr, comparisons, swaps


def visualize_selection_sort(arr):
    """
    Visual step-by-step
    """
    print("Selection Sort Visualization:")
    print("=" * 50)
    print("Initial:", arr, "\n")

    n = len(arr)

    for i in range(n):
        min_idx = i

        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        print("Step", i + 1, ":")
        print("  Sorted part:  ", arr[:i])
        print("  Current index:", i, "value:", arr[i])
        print("  Minimum found:", arr[min_idx], "at index", min_idx)

        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            print("  Swapped!")
        else:
            print("  No swap needed")

        print("  Array now:    ", arr)
        print()

    print("=" * 50)
    print("Final:", arr)


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 50)
    print("TEST 1: Basic Selection Sort")
    print("=" * 50)
    arr1 = [64, 25, 12, 22, 11]
    print("Original:", arr1)
    result1 = selectionSort(arr1.copy())
    print("Sorted:  ", result1)

    print("\n" + "=" * 50)
    print("TEST 2: Detailed Step-by-Step")
    print("=" * 50)
    arr2 = [5, 2, 4, 1, 3]
    selectionSort_detailed(arr2.copy())

    print("\n" + "=" * 50)
    print("TEST 3: Descending Order")
    print("=" * 50)
    arr3 = [5, 2, 4, 6, 1, 3]
    print("Original:  ", arr3)
    result3 = selectionSort_descending(arr3.copy())
    print("Descending:", result3)

    print("\n" + "=" * 50)
    print("TEST 4: With Statistics")
    print("=" * 50)
    arr4 = [64, 25, 12, 22, 11]
    original = arr4.copy()
    result4, comps, swaps = selectionSort_with_stats(arr4.copy())
    print("Original:   ", original)
    print("Sorted:     ", result4)
    print("Comparisons:", comps)
    print("Swaps:      ", swaps)

    print("\n" + "=" * 50)
    print("TEST 5: Visualization")
    print("=" * 50)
    arr5 = [4, 3, 2, 1]
    visualize_selection_sort(arr5.copy())

    print("\n" + "=" * 50)
    print("TEST 6: Edge Cases")
    print("=" * 50)

    # Single element
    print("Single element [5]:", selectionSort([5]))

    # Already sorted
    print("Already sorted [1,2,3,4,5]:", selectionSort([1,2,3,4,5]))

    # Reverse sorted
    print("Reverse sorted [5,4,3,2,1]:", selectionSort([5,4,3,2,1]))

    # All same
    print("All same [3,3,3,3]:", selectionSort([3,3,3,3]))

    # Empty
    print("Empty []:", selectionSort([]))

    # Two elements
    print("Two elements [2,1]:", selectionSort([2,1]))

    print("\n" + "=" * 50)
    print("TEST 7: Strings")
    print("=" * 50)
    strings = ["banana", "apple", "cherry", "date"]
    print("Original:", strings)
    selectionSort(strings)
    print("Sorted:  ", strings)


# In[ ]:




