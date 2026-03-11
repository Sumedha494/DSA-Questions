#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def insertionSort(arr):
    """
    Basic Insertion Sort
    Time Complexity: O(n²) - Worst/Average case
                    O(n) - Best case (already sorted)
    Space Complexity: O(1)
    """
    n = len(arr)

    # 1st element ko sorted maan lo
    for i in range(1, n):
        key = arr[i]  # Current element jo insert karna hai
        j = i - 1

        # Key se bade elements ko aage shift karo
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        # Key ko sahi position pe insert karo
        arr[j + 1] = key

    return arr


def insertionSort_detailed(arr):
    """
    Step-by-step explanation ke saath
    """
    print(f"Original Array: {arr}")
    print("=" * 60)

    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        print(f"\nPass {i}: Inserting {key}")
        print(f"Before: {arr}")
        print(f"Sorted part: {arr[:i]}, Unsorted: {arr[i:]}")

        # Shift elements
        shifts = 0
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
            shifts += 1

        arr[j + 1] = key

        print(f"After:  {arr}")
        print(f"Shifts made: {shifts}")
        print(f"Inserted {key} at index {j+1}")

    print("\n" + "=" * 60)
    print(f"Final Sorted Array: {arr}")
    return arr


def insertionSort_descending(arr):
    """
    Descending order mein sort karna
    """
    n = len(arr)

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        # Chote elements ko aage shift karo (descending ke liye)
        while j >= 0 and arr[j] < key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr


def insertionSort_recursive(arr, n=None):
    """
    Recursive approach
    """
    if n is None:
        n = len(arr)

    # Base case
    if n <= 1:
        return arr

    # Pehle n-1 elements ko sort karo
    insertionSort_recursive(arr, n - 1)

    # Last element ko sahi position pe insert karo
    key = arr[n - 1]
    j = n - 2

    while j >= 0 and arr[j] > key:
        arr[j + 1] = arr[j]
        j -= 1

    arr[j + 1] = key

    return arr


def binaryInsertionSort(arr):
    """
    Binary Search use karke sahi position dhundna
    Time Complexity: O(n²) - shifting still takes O(n)
    But comparisons reduce to O(n log n)
    """
    def binarySearch(arr, key, start, end):
        """Binary search se insertion position dhundna"""
        while start <= end:
            mid = (start + end) // 2

            if arr[mid] == key:
                return mid + 1
            elif arr[mid] < key:
                start = mid + 1
            else:
                end = mid - 1

        return start

    n = len(arr)

    for i in range(1, n):
        key = arr[i]

        # Binary search se position dhundho
        pos = binarySearch(arr, key, 0, i - 1)

        # Elements ko shift karo
        j = i - 1
        while j >= pos:
            arr[j + 1] = arr[j]
            j -= 1

        arr[pos] = key

    return arr


def insertionSort_with_stats(arr):
    """
    Sorting statistics ke saath
    """
    n = len(arr)
    comparisons = 0
    swaps = 0

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        while j >= 0:
            comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                swaps += 1
                j -= 1
            else:
                break

        arr[j + 1] = key

    return arr, comparisons, swaps


def visualize_insertion_sort(arr):
    """
    Visual representation
    """
    n = len(arr)

    print("Insertion Sort Visualization:")
    print("=" * 60)
    print(f"Initial: {arr}\n")

    for i in range(1, n):
        key = arr[i]
        j = i - 1

        # Visual representation
        print(f"Step {i}: Inserting {key}")

        # Show sorted and unsorted parts
        sorted_part = arr[:i]
        current = [key]
        unsorted = arr[i+1:] if i+1 < n else []

        print(f"  Sorted:   {sorted_part}")
        print(f"  Current:  {current}")
        print(f"  Unsorted: {unsorted}")

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

        print(f"  Result:   {arr}")
        print()


# Comparison with other sorting algorithms
def compare_sorts(arr):
    """
    Different sorting algorithms ki comparison
    """
    import time
    import copy

    print("Comparing Sorting Algorithms:")
    print("=" * 60)

    # Insertion Sort
    arr1 = copy.deepcopy(arr)
    start = time.time()
    insertionSort(arr1)
    time1 = time.time() - start

    # Built-in sort
    arr2 = copy.deepcopy(arr)
    start = time.time()
    arr2.sort()
    time2 = time.time() - start

    # Binary Insertion Sort
    arr3 = copy.deepcopy(arr)
    start = time.time()
    binaryInsertionSort(arr3)
    time3 = time.time() - start

    print(f"Array size: {len(arr)}")
    print(f"Insertion Sort:        {time1:.6f}s")
    print(f"Binary Insertion Sort: {time3:.6f}s")
    print(f"Built-in Sort:         {time2:.6f}s")


# Test Cases
if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Insertion Sort")
    print("=" * 60)
    arr1 = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {arr1}")
    result1 = insertionSort(arr1.copy())
    print(f"Sorted:   {result1}\n")

    print("=" * 60)
    print("TEST 2: Detailed Step-by-Step")
    print("=" * 60)
    arr2 = [5, 2, 4, 6, 1, 3]
    insertionSort_detailed(arr2.copy())

    print("\n" + "=" * 60)
    print("TEST 3: Descending Order")
    print("=" * 60)
    arr3 = [5, 2, 4, 6, 1, 3]
    print(f"Original:   {arr3}")
    result3 = insertionSort_descending(arr3.copy())
    print(f"Descending: {result3}\n")

    print("=" * 60)
    print("TEST 4: Recursive Insertion Sort")
    print("=" * 60)
    arr4 = [12, 11, 13, 5, 6]
    print(f"Original:  {arr4}")
    result4 = insertionSort_recursive(arr4.copy())
    print(f"Recursive: {result4}\n")

    print("=" * 60)
    print("TEST 5: Binary Insertion Sort")
    print("=" * 60)
    arr5 = [37, 23, 0, 17, 12, 72, 31]
    print(f"Original: {arr5}")
    result5 = binaryInsertionSort(arr5.copy())
    print(f"Binary:   {result5}\n")

    print("=" * 60)
    print("TEST 6: With Statistics")
    print("=" * 60)
    arr6 = [5, 2, 4, 6, 1, 3]
    result6, comps, swaps = insertionSort_with_stats(arr6.copy())
    print(f"Original:     {[5, 2, 4, 6, 1, 3]}")
    print(f"Sorted:       {result6}")
    print(f"Comparisons:  {comps}")
    print(f"Swaps:        {swaps}\n")

    print("=" * 60)
    print("TEST 7: Visualization")
    print("=" * 60)
    arr7 = [4, 3, 2, 1]
    visualize_insertion_sort(arr7.copy())

    print("=" * 60)
    print("TEST 8: Edge Cases")
    print("=" * 60)
    edge_cases = [
        ([1], "Single element"),
        ([1, 2, 3, 4, 5], "Already sorted"),
        ([5, 4, 3, 2, 1], "Reverse sorted"),
        ([3, 3, 3, 3], "All same"),
        ([], "Empty array"),
    ]

    for arr, desc in edge_cases:
        original = arr.copy()
        result = insertionSort(arr.copy()) if arr else arr
        print(f"{desc:20s}: {original} → {result}")

    print("\n" + "=" * 60)
    print("TEST 9: Performance Comparison")
    print("=" * 60)
    import random
    test_arr = [random.randint(1, 100) for _ in range(100)]
    compare_sorts(test_arr)

    print("\n" + "=" * 60)
    print("TEST 10: String Sorting")
    print("=" * 60)
    strings = ["banana", "apple", "cherry", "date", "elderberry"]
    print(f"Original: {strings}")
    insertionSort(strings)
    print(f"Sorted:   {strings}")

