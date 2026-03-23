#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def moveZeroes(arr):
    """
    Move all zeroes to end, maintain order of non-zeroes

    Two Pointer Approach (Optimal)
    Time: O(n), Space: O(1)
    """
    if not arr:
        return arr

    # Pointer for non-zero position
    j = 0

    # Move non-zero elements forward
    for i in range(len(arr)):
        if arr[i] != 0:
            arr[j] = arr[i]
            j += 1

    # Fill remaining with zeroes
    while j < len(arr):
        arr[j] = 0
        j += 1

    return arr


def moveZeroes_swap(arr):
    """
    Using swap method
    Time: O(n), Space: O(1)
    """
    if not arr:
        return arr

    j = 0  # Position for next non-zero

    for i in range(len(arr)):
        if arr[i] != 0:
            arr[i], arr[j] = arr[j], arr[i]
            j += 1

    return arr


def moveZeroes_detailed(arr):
    """
    Step-by-step explanation
    """
    print("Move Zeroes to End")
    print("=" * 50)
    print("Original:", arr)
    print()

    if not arr:
        return arr

    j = 0

    print("Step-by-Step:")
    print("-" * 50)

    for i in range(len(arr)):
        print("i=" + str(i) + ", j=" + str(j) + ", arr[i]=" + str(arr[i]), end=" ")

        if arr[i] != 0:
            if i != j:
                arr[i], arr[j] = arr[j], arr[i]
                print("-> Swap arr[" + str(i) + "] with arr[" + str(j) + "]")
            else:
                print("-> No swap needed")
            j += 1
        else:
            print("-> Zero, skip")

        print("   Array:", arr)

    print()
    print("=" * 50)
    print("Result:", arr)

    return arr


def moveZeroes_count(arr):
    """
    Count method (simple but not optimal swaps)
    Time: O(n), Space: O(1)
    """
    if not arr:
        return arr

    # Count zeroes
    zero_count = arr.count(0)

    # Remove all zeroes
    arr[:] = [x for x in arr if x != 0]

    # Add zeroes at end
    arr.extend([0] * zero_count)

    return arr


def moveZeroes_start(arr):
    """
    Move zeroes to START instead of end
    """
    if not arr:
        return arr

    j = len(arr) - 1

    for i in range(len(arr) - 1, -1, -1):
        if arr[i] != 0:
            arr[j] = arr[i]
            j -= 1

    while j >= 0:
        arr[j] = 0
        j -= 1

    return arr


def moveNegatives_end(arr):
    """
    Similar: Move negatives to end
    """
    if not arr:
        return arr

    j = 0

    for i in range(len(arr)):
        if arr[i] >= 0:
            arr[i], arr[j] = arr[j], arr[i]
            j += 1

    return arr


def visualize_move_zeroes(arr):
    """
    Visual representation
    """
    print("Move Zeroes Visualization")
    print("=" * 50)
    print("Original:", arr)
    print()

    j = 0

    for i in range(len(arr)):
        # Show pointers
        pointer_line = ""
        for k in range(len(arr)):
            if k == i and k == j:
                pointer_line += "i,j "
            elif k == i:
                pointer_line += " i  "
            elif k == j:
                pointer_line += " j  "
            else:
                pointer_line += "    "

        print("Array:", arr)
        print("      ", pointer_line)

        if arr[i] != 0:
            if i != j:
                arr[i], arr[j] = arr[j], arr[i]
                print("Action: Swap!")
            else:
                print("Action: Move j")
            j += 1
        else:
            print("Action: Skip zero")

        print()

    print("Final:", arr)


def moveZeroes_with_stats(arr):
    """
    Count operations
    """
    if not arr:
        return arr, 0, 0

    swaps = 0
    comparisons = 0
    j = 0

    for i in range(len(arr)):
        comparisons += 1

        if arr[i] != 0:
            if i != j:
                arr[i], arr[j] = arr[j], arr[i]
                swaps += 1
            j += 1

    return arr, comparisons, swaps


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 50)
    print("TEST 1: Basic Example")
    print("=" * 50)

    arr1 = [0, 1, 0, 3, 12]
    print("Input: ", arr1)
    moveZeroes(arr1)
    print("Output:", arr1)
    print("Expected: [1, 3, 12, 0, 0]")

    print("\n" + "=" * 50)
    print("TEST 2: Detailed Step-by-Step")
    print("=" * 50)

    arr2 = [0, 1, 0, 3, 12]
    moveZeroes_detailed(arr2.copy())

    print("\n" + "=" * 50)
    print("TEST 3: Swap Method")
    print("=" * 50)

    arr3 = [0, 1, 0, 3, 12]
    print("Input: ", arr3)
    moveZeroes_swap(arr3)
    print("Output:", arr3)

    print("\n" + "=" * 50)
    print("TEST 4: Visualization")
    print("=" * 50)

    arr4 = [0, 1, 0, 3]
    visualize_move_zeroes(arr4.copy())

    print("\n" + "=" * 50)
    print("TEST 5: Edge Cases")
    print("=" * 50)

    edge_cases = [
        ([0], "Single zero"),
        ([1], "Single non-zero"),
        ([0, 0, 0], "All zeroes"),
        ([1, 2, 3], "No zeroes"),
        ([1, 0], "Zero at end"),
        ([0, 1], "Zero at start"),
        ([], "Empty array"),
    ]

    for arr, desc in edge_cases:
        original = arr.copy()
        result = moveZeroes(arr.copy())
        print(desc.ljust(20), ":", original, "->", result)

    print("\n" + "=" * 50)
    print("TEST 6: With Statistics")
    print("=" * 50)

    arr6 = [0, 1, 0, 3, 12]
    original = arr6.copy()
    result, comps, swaps = moveZeroes_with_stats(arr6)

    print("Array:      ", original)
    print("Result:     ", result)
    print("Comparisons:", comps)
    print("Swaps:      ", swaps)

    print("\n" + "=" * 50)
    print("TEST 7: Move Zeroes to Start")
    print("=" * 50)

    arr7 = [1, 0, 2, 0, 3]
    print("Input: ", arr7)
    moveZeroes_start(arr7)
    print("Output:", arr7)
    print("(Zeroes at start)")

    print("\n" + "=" * 50)
    print("TEST 8: Move Negatives to End")
    print("=" * 50)

    arr8 = [1, -2, 3, -4, 5]
    print("Input: ", arr8)
    moveNegatives_end(arr8)
    print("Output:", arr8)

    print("\n" + "=" * 50)
    print("TEST 9: Large Array")
    print("=" * 50)

    import random
    arr9 = [random.choice([0, 1, 2, 3]) for _ in range(15)]
    print("Input: ", arr9)
    moveZeroes(arr9)
    print("Output:", arr9)

    print("\n" + "=" * 50)
    print("TEST 10: Compare Methods")
    print("=" * 50)

    import time

    test_arr = [0, 1, 0, 2, 0, 3, 0, 4] * 1000

    # Method 1: Two pointer
    arr_copy1 = test_arr.copy()
    start = time.time()
    moveZeroes(arr_copy1)
    time1 = time.time() - start

    # Method 2: Swap
    arr_copy2 = test_arr.copy()
    start = time.time()
    moveZeroes_swap(arr_copy2)
    time2 = time.time() - start

    # Method 3: Count
    arr_copy3 = test_arr.copy()
    start = time.time()
    moveZeroes_count(arr_copy3)
    time3 = time.time() - start

    print("Array size:", len(test_arr))
    print("Two Pointer:", round(time1, 6), "s")
    print("Swap Method:", round(time2, 6), "s")
    print("Count Method:", round(time3, 6), "s")

    print("\n" + "=" * 50)
    print("ALGORITHM SUMMARY")
    print("=" * 50)
    print("""
Move Zeroes to End

Problem: [0, 1, 0, 3, 12] -> [1, 3, 12, 0, 0]

Approach 1: Two Pointer (Overwrite)
  - j tracks position for non-zero
  - Move all non-zeroes to front
  - Fill rest with zeroes
  Time: O(n), Space: O(1)

Approach 2: Two Pointer (Swap)
  - Swap non-zero with position j
  - Maintains relative order
  Time: O(n), Space: O(1)

Approach 3: Count Method
  - Count zeroes
  - Remove zeroes
  - Add zeroes at end
  Time: O(n), Space: O(1)*

*Creates new list internally

LeetCode #283: Move Zeroes
    """)

    print("\n" + "=" * 50)
    print("VISUAL EXPLANATION")
    print("=" * 50)
    print("""
Array: [0, 1, 0, 3, 12]
        j
        i

Step 1: arr[0]=0, skip
        [0, 1, 0, 3, 12]
        j
           i

Step 2: arr[1]=1, swap with j
        [1, 0, 0, 3, 12]
           j
              i

Step 3: arr[2]=0, skip
        [1, 0, 0, 3, 12]
           j
                 i

Step 4: arr[3]=3, swap with j
        [1, 3, 0, 0, 12]
              j
                    i

Step 5: arr[4]=12, swap with j
        [1, 3, 12, 0, 0]
                  j

Final: [1, 3, 12, 0, 0]
    """)

