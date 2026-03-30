#!/usr/bin/env python
# coding: utf-8

# In[2]:


def maxProductCount(arr):
    """
    Find count of maximum product subarray
    Time: O(n²), Space: O(1)
    """
    if not arr:
        return 0

    n = len(arr)
    max_product = arr[0]

    # Find maximum product
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            if product > max_product:
                max_product = product

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
    print("=" * 50)
    print("Array:", arr)
    print()

    if not arr:
        return 0

    n = len(arr)

    # Find all subarray products
    print("All Subarrays and Products:")
    print("-" * 40)

    all_products = []

    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= arr[j]
            subarray = arr[i:j+1]
            all_products.append((subarray, product))
            print(" ", subarray, "=", product)

    print()

    # Find maximum
    max_product = max(p for _, p in all_products)
    print("Maximum Product:", max_product)
    print()

    # Count
    print("Subarrays with max product:")
    count = 0
    for subarray, product in all_products:
        if product == max_product:
            count += 1
            print(" ", subarray, "=", product, "✓")

    print()
    print("=" * 50)
    print("Count:", count)

    return count


def maxProductSubarray(arr):
    """
    Find maximum product (Kadane style)
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0

    max_prod = arr[0]
    max_end = arr[0]
    min_end = arr[0]

    for i in range(1, len(arr)):
        if arr[i] < 0:
            max_end, min_end = min_end, max_end


