#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def bubbleSort(arr):
    """
    Basic Bubble Sort
    Time Complexity: O(n²) - Worst/Average case
                    O(n) - Best case (optimized version)
    Space Complexity: O(1)
    """
    n = len(arr)

    # n-1 passes
    for i in range(n):
        # Har pass mein largest element end mein chala jata hai
        for j in range(0, n - i - 1):
            # Adjacent elements compare karo
            if arr[j] > arr[j + 1]:
                # Swap karo
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr


def bubbleSort_optimized(arr):
    """
    Optimized Bubble Sort with early termination
    Agar koi swap nahi hua matlab array sorted hai
    """
    n = len(arr)

    for i in range(n):
        swapped = False  # Flag to check if any swap happened

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # Agar koi swap nahi hua, array sorted hai
        if not swapped:
            print("Array sorted early at pass", i + 1)
            break

    return arr


def bubbleSort_detailed(arr):
    """
    Step-by-step explanation ke saath
    """
    print("Original Array:", arr)
    print("=" * 60)

    n = len(arr)

    for i in range(n):
        print("\nPass", i + 1, ":")
        swapped = False

        for j in range(0, n - i - 1):
            print("  Comparing", arr[j], "and", arr[j + 1], end=" ")

            if arr[j] > arr[j + 1]:
                print("-> Swap needed")
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
            else:
                print("-> No swap")

        print("After pass", i + 1, ":", arr)
        print("Largest element", arr[n - i - 1], "is now at position", n - i - 1)

        if not swapped:
            print("\nNo swaps made! Array is sorted.")
            break

    print("\n" + "=" * 60)
    print("Final Sorted Array:", arr)
    return arr


def bubbleSort_descending(arr):
    """
    Descending order mein sort
    """
    n = len(arr)

    for i in range(n):
        for j in range(0, n - i - 1):
            # Chote element ko aage bhejo
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr


def bubbleSort_with_stats(arr):
    """
    Statistics track karo - comparisons and swaps
    """
    n = len(arr)
    comparisons = 0
    swaps = 0

    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1

            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1

    return arr, comparisons, swaps


def bubbleSort_recursive(arr, n=None):
    """
    Recursive Bubble Sort
    """
    if n is None:
        n = len(arr)

    # Base case
    if n == 1:
        return arr

    # Ek pass kar lo - largest element end mein chala jayega
    for i in range(n - 1):
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]

    # Recursively remaining array ko sort karo
    return bubbleSort_recursive(arr, n - 1)


def visualize_bubble_sort(arr):
    """
    Visual representation with step-by-step animation
    """
    print("Bubble Sort Visualization:")
    print("=" * 60)
    print("Initial:", arr, "\n")

    n = len(arr)

    for i in range(n):
        print("Pass", i + 1, ":")
        swapped = False

        for j in range(0, n - i - 1):
            # Show current comparison
            display = []
            for k in range(len(arr)):
                if k == j:
                    display.append("[" + str(arr[k]) + "]")
                elif k == j + 1:
                    display.append("[" + str(arr[k]) + "]")
                else:
                    display.append(" " + str(arr[k]) + " ")

            print("  Comparing:", " ".join(display), end=" ")

            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                print("→ SWAP!")
                swapped = True
            else:
                print("→ OK")

        print("  Result:   ", arr)
        print()

        if not swapped:
            print("Array is sorted!")
            break

    print("=" * 60)
    print("Final:", arr)


def cocktail_shaker_sort(arr):
    """
    Bidirectional Bubble Sort (Cocktail Shaker Sort)
    Left to right aur right to left dono direction mein bubble karo
    """
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False

        # Left to right pass
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        if not swapped:
            break

        swapped = False
        end -= 1

        # Right to left pass
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        start += 1

    return arr


def compare_bubble_versions():
    """
    Normal vs Optimized Bubble Sort comparison
    """
    import time
    import random

    print("Comparing Bubble Sort Versions:")
    print("=" * 60)

    # Test with already sorted array
    arr_sorted = list(range(1000))

    # Normal bubble sort
    arr1 = arr_sorted.copy()
    start = time.time()
    bubbleSort(arr1)
    time_normal = time.time() - start

    # Optimized bubble sort
    arr2 = arr_sorted.copy()
    start = time.time()
    bubbleSort_optimized(arr2)
    time_optimized = time.time() - start

    print("On Already Sorted Array (1000 elements):")
    print("Normal Bubble Sort:    ", round(time_normal, 6), "seconds")
    print("Optimized Bubble Sort: ", round(time_optimized, 6), "seconds")
    print("Speedup:", round(time_normal / time_optimized, 2), "x faster!")


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Bubble Sort")
    print("=" * 60)
    arr1 = [64, 34, 25, 12, 22, 11, 90]
    print("Original:", arr1)
    result1 = bubbleSort(arr1.copy())
    print("Sorted:  ", result1)

    print("\n" + "=" * 60)
    print("TEST 2: Optimized Bubble Sort")
    print("=" * 60)
    arr2 = [1, 2, 3, 5, 4]  # Almost sorted
    print("Original:", arr2)
    result2 = bubbleSort_optimized(arr2.copy())
    print("Sorted:  ", result2)

    print("\n" + "=" * 60)
    print("TEST 3: Detailed Step-by-Step")
    print("=" * 60)
    arr3 = [5, 1, 4, 2, 8]
    bubbleSort_detailed(arr3.copy())

    print("\n" + "=" * 60)
    print("TEST 4: Descending Order")
    print("=" * 60)
    arr4 = [5, 2, 4, 6, 1, 3]
    print("Original:  ", arr4)
    result4 = bubbleSort_descending(arr4.copy())
    print("Descending:", result4)

    print("\n" + "=" * 60)
    print("TEST 5: With Statistics")
    print("=" * 60)
    arr5 = [64, 34, 25, 12, 22, 11, 90]
    original = arr5.copy()
    result5, comps, swaps = bubbleSort_with_stats(arr5.copy())
    print("Original:   ", original)
    print("Sorted:     ", result5)
    print("Comparisons:", comps)
    print("Swaps:      ", swaps)

    print("\n" + "=" * 60)
    print("TEST 6: Recursive Bubble Sort")
    print("=" * 60)
    arr6 = [12, 11, 13, 5, 6]
    print("Original: ", arr6)
    result6 = bubbleSort_recursive(arr6.copy())
    print("Recursive:", result6)

    print("\n" + "=" * 60)
    print("TEST 7: Visualization")
    print("=" * 60)
    arr7 = [5, 3, 8, 4, 2]
    visualize_bubble_sort(arr7.copy())

    print("\n" + "=" * 60)
    print("TEST 8: Cocktail Shaker Sort")
    print("=" * 60)
    arr8 = [5, 1, 4, 2, 8, 0, 2]
    print("Original:       ", arr8)
    result8 = cocktail_shaker_sort(arr8.copy())
    print("Cocktail Sorted:", result8)

    print("\n" + "=" * 60)
    print("TEST 9: Edge Cases")
    print("=" * 60)

    test_cases = [
        ([1], "Single element"),
        ([1, 2, 3, 4, 5], "Already sorted"),
        ([5, 4, 3, 2, 1], "Reverse sorted"),
        ([3, 3, 3, 3], "All same"),
        ([], "Empty array"),
        ([2, 1], "Two elements"),
        ([1, 1, 2, 2, 3], "Duplicates"),
    ]

    for arr, desc in test_cases:
        original = arr.copy()
        result = bubbleSort(arr.copy()) if arr else arr
        print(desc.ljust(20), ":", original, "->", result)

    print("\n" + "=" * 60)
    print("TEST 10: Different Data Types")
    print("=" * 60)

    # Strings
    strings = ["banana", "apple", "cherry", "date"]
    print("Strings:", strings)
    bubbleSort(strings)
    print("Sorted: ", strings)

    # Floats
    floats = [3.14, 1.41, 2.71, 0.99]
    print("\nFloats: ", floats)
    bubbleSort(floats)
    print("Sorted: ", floats)

    # Characters
    chars = ['d', 'a', 'c', 'b']
    print("\nChars:  ", chars)
    bubbleSort(chars)
    print("Sorted: ", chars)

    print("\n" + "=" * 60)
    print("TEST 11: Performance Analysis")
    print("=" * 60)

    # Best case - already sorted
    arr_best = list(range(50))
    _, comps_best, swaps_best = bubbleSort_with_stats(arr_best.copy())

    # Worst case - reverse sorted
    arr_worst = list(range(50, 0, -1))
    _, comps_worst, swaps_worst = bubbleSort_with_stats(arr_worst.copy())

    # Average case - random
    import random
    arr_avg = [random.randint(1, 100) for _ in range(50)]
    _, comps_avg, swaps_avg = bubbleSort_with_stats(arr_avg.copy())

    print("Array size: 50\n")
    print("Best Case (Sorted):")
    print("  Comparisons:", comps_best)
    print("  Swaps:      ", swaps_best)

    print("\nWorst Case (Reverse Sorted):")
    print("  Comparisons:", comps_worst)
    print("  Swaps:      ", swaps_worst)

    print("\nAverage Case (Random):")
    print("  Comparisons:", comps_avg)
    print("  Swaps:      ", swaps_avg)

    print("\n" + "=" * 60)
    print("TEST 12: Optimization Impact")
    print("=" * 60)
    compare_bubble_versions()


# In[ ]:




