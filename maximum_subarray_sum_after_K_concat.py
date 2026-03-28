#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def maxSumAfterKConcat(arr, k):
    """
    Maximum Subarray Sum after K Concatenations

    Time: O(n), Space: O(n)
    """
    if not arr:
        return 0

    total_sum = sum(arr)

    def kadane(array):
        max_sum = array[0]
        current = array[0]

        for i in range(1, len(array)):
            current = max(array[i], current + array[i])
            max_sum = max(max_sum, current)

        return max_sum

    if k == 1:
        return kadane(arr)

    # Kadane on doubled array
    doubled_kadane = kadane(arr + arr)

    if k == 2:
        return doubled_kadane

    # k >= 3
    if total_sum > 0:
        return doubled_kadane + (k - 2) * total_sum
    else:
        return doubled_kadane


def maxSumAfterKConcat_detailed(arr, k):
    """
    Step-by-step explanation
    """
    print("Maximum Subarray Sum After K Concatenations")
    print("=" * 60)
    print("Array:", arr)
    print("K:", k)
    print()

    if not arr:
        return 0

    total_sum = sum(arr)
    print("Total sum:", total_sum)

    def kadane(array):
        max_sum = array[0]
        current = array[0]
        for i in range(1, len(array)):
            current = max(array[i], current + array[i])
            max_sum = max(max_sum, current)
        return max_sum

    single = kadane(arr)
    print("Kadane on single array:", single)

    if k == 1:
        print("Answer:", single)
        return single

    doubled = kadane(arr + arr)
    print("Kadane on doubled array:", doubled)

    if k == 2:
        print("Answer:", doubled)
        return doubled

    print()
    if total_sum > 0:
        result = doubled + (k - 2) * total_sum
        print("total_sum > 0")
        print("Formula: doubled + (k-2) * total_sum")
        print("       :", doubled, "+", k-2, "*", total_sum, "=", result)
    else:
        result = doubled
        print("total_sum <= 0, extra copies won't help")
        print("Answer:", result)

    return result


def kadane_simple(arr):
    """
    Standard Kadane's algorithm
    """
    max_sum = arr[0]
    current = arr[0]

    for i in range(1, len(arr)):
        current = max(arr[i], current + arr[i])
        max_sum = max(max_sum, current)

    return max_sum


def maxSumAfterKConcat_brute(arr, k):
    """
    Brute force - only for small k
    """
    if not arr:
        return 0

    concat = arr * k
    return kadane_simple(concat)


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    # Test case 1
    arr1 = [1, 2]
    k1 = 3
    result1 = maxSumAfterKConcat(arr1, k1)
    print("arr:", arr1, ", k:", k1, "->", result1, "(Expected: 9)")

    # Test case 2
    arr2 = [1, -2, 1]
    k2 = 5
    result2 = maxSumAfterKConcat(arr2, k2)
    print("arr:", arr2, ", k:", k2, "->", result2, "(Expected: 2)")

    # Test case 3
    arr3 = [-1, -2]
    k3 = 7
    result3 = maxSumAfterKConcat(arr3, k3)
    print("arr:", arr3, ", k:", k3, "->", result3, "(Expected: -1)")

    # Test case 4
    arr4 = [1, 2, 3]
    k4 = 2
    result4 = maxSumAfterKConcat(arr4, k4)
    print("arr:", arr4, ", k:", k4, "->", result4, "(Expected: 12)")

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    maxSumAfterKConcat_detailed([1, 2], 3)

    print("\n" + "=" * 60)
    print("TEST 3: Negative Total Sum")
    print("=" * 60)

    maxSumAfterKConcat_detailed([1, -2, 1], 5)

    print("\n" + "=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)

    # Single element
    print("Single element [5], k=3:", maxSumAfterKConcat([5], 3))

    # All negative
    print("All negative [-1,-2], k=3:", maxSumAfterKConcat([-1, -2], 3))

    # k = 1
    print("k=1 [1,2,3]:", maxSumAfterKConcat([1, 2, 3], 1))

    # Sum = 0
    print("Sum=0 [1,-1], k=5:", maxSumAfterKConcat([1, -1], 5))

    print("\n" + "=" * 60)
    print("TEST 5: Compare Brute vs Optimal")
    print("=" * 60)

    arr5 = [1, 2, -3, 4]
    for k in range(1, 6):
        brute = maxSumAfterKConcat_brute(arr5, k)
        optimal = maxSumAfterKConcat(arr5, k)
        status = "OK" if brute == optimal else "FAIL"
        print("k=" + str(k) + ": Brute=" + str(brute) + ", Optimal=" + str(optimal), status)

    print("\n" + "=" * 60)
    print("TEST 6: Large K")
    print("=" * 60)

    arr6 = [1, 2, 3]
    k6 = 100000
    result6 = maxSumAfterKConcat(arr6, k6)
    print("arr:", arr6)
    print("k:", k6)
    print("Result:", result6)
    print("Expected:", 6 * k6)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("K = 1: kadane(arr)")
    print("K = 2: kadane(arr + arr)")
    print("K >= 3:")
    print("  if sum > 0: kadane(arr+arr) + (k-2)*sum")
    print("  else:       kadane(arr+arr)")
    print()
    print("Time: O(n)")
    print("Space: O(n)")

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print()
    print("arr = [1, 2], k = 3")
    print("total_sum = 3 (positive)")
    print()
    print("Concatenated: [1, 2, 1, 2, 1, 2]")
    print("               |------sum-------|")
    print("               = 1+2+1+2+1+2 = 9")
    print()
    print("Formula:")
    print("  kadane([1,2,1,2]) + (3-2) * 3")
    print("  = 6 + 1 * 3")
    print("  = 9")

