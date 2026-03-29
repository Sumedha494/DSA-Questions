#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def maxProductCount(arr):
    """
    Find count of maximum product subarray

    First find max product, then count subarrays with that product
    Time: O(n²), Space: O(1)
    """
    if not arr:
        return 0

    n = len(arr)
    max_product = arr[0]
    count = 0

    # Find maximum product
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            if product > max_product:
                max_product = product

    # Count subarrays with max product
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            if product == max_product:
                count += 1

    return count


def maxProductCount_optimized(arr):
    """
    Optimized approach
    Find max product using Kadane's style, then count
    Time: O(n) for max product + O(n²) for count
    """
    if not arr:
        return 0

    n = len(arr)

    # Find max product using modified Kadane
    max_product = arr[0]
    max_ending = arr[0]
    min_ending = arr[0]

    for i in range(1, n):
        if arr[i] < 0:
            max_ending, min_ending = min_ending, max_ending

        max_ending = max(arr[i], max_ending * arr[i])
        min_ending = min(arr[i], min_ending * arr[i])

        max_product = max(max_product, max_ending)

    # Count subarrays with max product
    count = 0
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            if product == max_product:
                count += 1

    return count, max_product


def maxProductCount_detailed(arr):
    """
    Step-by-step explanation
    """
    print("Maximum Product Count")
    print("=" * 60)
    print("Array:", arr)
    print()

    if not arr:
        return 0

    n = len(arr)

    # Find all subarray products
    print("Step 1: Find all subarray products")
    print("-" * 40)

    all_products = []

    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            subarray = arr[i:j+1]
            all_products.append((subarray, product))
            print("  ", subarray, "->", product)

    print()

    # Find maximum
    max_product = max(p for _, p in all_products)
    print("Step 2: Maximum product:", max_product)
    print()

    # Count
    print("Step 3: Count subarrays with max product")
    print("-" * 40)

    count = 0
    for subarray, product in all_products:
        if product == max_product:
            count += 1
            print("  ", subarray, "=", product, "✓")

    print()
    print("=" * 60)
    print("Maximum Product:", max_product)
    print("Count:", count)

    return count


def maxProductSubarray(arr):
    """
    Find maximum product subarray (Kadane style)
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0

    max_product = arr[0]
    max_ending = arr[0]
    min_ending = arr[0]

    for i in range(1, len(arr)):
        if arr[i] < 0:
            max_ending, min_ending = min_ending, max_ending

        max_ending = max(arr[i], max_ending * arr[i])
        min_ending = min(arr[i], min_ending * arr[i])

        max_product = max(max_product, max_ending)

    return max_product


def maxProductWithSubarray(arr):
    """
    Return max product and the actual subarray
    """
    if not arr:
        return 0, []

    n = len(arr)
    max_product = arr[0]
    max_subarray = [arr[0]]

    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            if product > max_product:
                max_product = product
                max_subarray = arr[i:j+1]

    return max_product, max_subarray


def countAllProducts(arr):
    """
    Count frequency of each product
    """
    if not arr:
        return {}

    n = len(arr)
    product_count = {}

    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            product_count[product] = product_count.get(product, 0) + 1

    return product_count


def visualizeProducts(arr):
    """
    Visual representation of all products
    """
    print("Product Visualization")
    print("=" * 60)
    print("Array:", arr)
    print()

    n = len(arr)
    products = []

    print("All Subarrays and Products:")
    print("-" * 40)

    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            subarray = arr[i:j+1]
            products.append((subarray, product))

    # Sort by product value
    products.sort(key=lambda x: x[1], reverse=True)

    for subarray, product in products:
        bar = "*" * min(abs(product), 20)
        if product < 0:
            bar = "-" * min(abs(product), 20)
        print("  ", subarray, "=", product, bar)

    print()

    max_product = max(p for _, p in products)
    count = sum(1 for _, p in products if p == max_product)

    print("Maximum Product:", max_product)
    print("Count:", count)

    return count


def maxProductCountNegatives(arr):
    """
    Handle arrays with zeros and negatives
    """
    if not arr:
        return 0

    n = len(arr)
    max_product = float('-inf')

    # Find max product
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            max_product = max(max_product, product)

    # Handle all zeros case
    if max_product == float('-inf'):
        max_product = 0

    # Count
    count = 0
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            if product == max_product:
                count += 1

    return count, max_product


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    # Test 1: Simple array
    arr1 = [2, 3, 2]
    count1 = maxProductCount(arr1)
    print("Array:", arr1)
    print("Max Product Count:", count1)
    print("(Max product = 12, appears once)")

    print()

    # Test 2: Array with duplicates
    arr2 = [2, 2, 2]
    count2 = maxProductCount(arr2)
    print("Array:", arr2)
    print("Max Product Count:", count2)
    print("(Max product = 8, appears once)")

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    arr3 = [1, 2, 3]
    maxProductCount_detailed(arr3)

    print("\n" + "=" * 60)
    print("TEST 3: With Optimized Function")
    print("=" * 60)

    arr4 = [2, 3, -2, 4]
    count4, max_prod4 = maxProductCount_optimized(arr4)
    print("Array:", arr4)
    print("Max Product:", max_prod4)
    print("Count:", count4)

    print("\n" + "=" * 60)
    print("TEST 4: Get Actual Subarray")
    print("=" * 60)

    arr5 = [2, 3, -2, 4]
    max_prod5, subarray5 = maxProductWithSubarray(arr5)
    print("Array:", arr5)
    print("Max Product:", max_prod5)
    print("Subarray:", subarray5)

    print("\n" + "=" * 60)
    print("TEST 5: Visualization")
    print("=" * 60)

    arr6 = [2, 3, 2]
    visualizeProducts(arr6)

    print("\n" + "=" * 60)
    print("TEST 6: Edge Cases")
    print("=" * 60)

    # Single element
    arr7 = [5]
    count7 = maxProductCount(arr7)
    print("Single element [5]:", count7)

    # All ones
    arr8 = [1, 1, 1]
    count8 = maxProductCount(arr8)
    print("All ones [1,1,1]:", count8)

    # With zero
    arr9 = [0, 2, 3]
    count9, max9 = maxProductCountNegatives(arr9)
    print("With zero [0,2,3]: max=", max9, ", count=", count9)

    # All negative
    arr10 = [-1, -2, -3]
    count10, max10 = maxProductCountNegatives(arr10)
    print("All negative [-1,-2,-3]: max=", max10, ", count=", count10)

    print("\n" + "=" * 60)
    print("TEST 7: Count All Products")
    print("=" * 60)

    arr11 = [2, 3, 2]
    products = countAllProducts(arr11)
    print("Array:", arr11)
    print("Product frequencies:")
    for prod, cnt in sorted(products.items(), reverse=True):
        print("  Product", prod, ":", cnt, "times")

    print("\n" + "=" * 60)
    print("TEST 8: Negative Numbers")
    print("=" * 60)

    arr12 = [-2, 3, -4]
    maxProductCount_detailed(arr12)

    print("\n" + "=" * 60)
    print("TEST 9: Multiple Max Products")
    print("=" * 60)

    arr13 = [2, 2, 2, 2]
    print("Array:", arr13)
    products13 = countAllProducts(arr13)
    max_prod = max(products13.keys())
    print("Max Product:", max_prod)
    print("Count:", products13[max_prod])

    print("\n" + "=" * 60)
    print("TEST 10: Large Array")
    print("=" * 60)

    arr14 = [1, 2, 3, 4, 5]
    count14, max14 = maxProductCount_optimized(arr14)
    print("Array:", arr14)
    print("Max Product:", max14)
    print("Count:", count14)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Maximum Product Count

Problem: Find count of subarrays with maximum product

Approach:
  1. Find maximum product (Kadane style or brute force)
  2. Count subarrays with that product

Time Complexity:
  - Brute force: O(n^2)
  - Max product: O(n) using Kadane

Space Complexity: O(1)

Key Points:
  - Handle negative numbers (flip max/min)
  - Handle zeros (break the product chain)
  - Two negative = positive product
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print("""
Array: [2, 3, 2]

All Subarrays:
  [2]       = 2
  [2,3]     = 6
  [2,3,2]   = 12  <- MAX
  [3]       = 3
  [3,2]     = 6
  [2]       = 2

Maximum Product: 12
Count: 1

Another Example: [1, 1, 1]

All Subarrays:
  [1]       = 1
  [1,1]     = 1
  [1,1,1]   = 1  <- MAX (tied)
  [1]       = 1
  [1,1]     = 1
  [1]       = 1

Maximum Product: 1
Count: 6 (all subarrays have product 1!)
    """)

    print("\n" + "=" * 60)
    print("HANDLING SPECIAL CASES")
    print("=" * 60)
    print("""
1. Negative Numbers:
   [-2, 3, -4]
   Products: -2, -6, 24, 3, -12, -4
   Max = 24 (two negatives make positive!)

2. Zeros:
   [0, 2, 3]
   Products: 0, 0, 0, 2, 6, 3
   Max = 6, Count = 1

3. All Negatives:
   [-1, -2]
   Products: -1, 2, -2
   Max = 2 (product of two negatives)

4. Single Element:
   [5]
   Max = 5, Count = 1
    """)

