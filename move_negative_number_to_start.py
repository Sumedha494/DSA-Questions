#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def move_negatives_to_start(arr):
    """
    Method 1: Two Pointer Approach (In-place)
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left = 0
    right = len(arr) - 1

    while left <= right:
        # Agar left negative hai aur right positive hai
        if arr[left] < 0 and arr[right] >= 0:
            left += 1
            right -= 1
        # Agar left positive hai aur right negative hai, swap karo
        elif arr[left] >= 0 and arr[right] < 0:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
        # Agar left negative hai
        elif arr[left] < 0:
            left += 1
        # Agar right positive hai
        else:
            right -= 1

    return arr

def move_negatives_method2(arr):
    """
    Method 2: Single Pointer Approach (In-place)
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    j = 0  # Negative numbers ki position track karne ke liye

    for i in range(len(arr)):
        if arr[i] < 0:
            # Swap karo negative number ko start mein
            arr[i], arr[j] = arr[j], arr[i]
            j += 1

    return arr

def move_negatives_method3(arr):
    """
    Method 3: Extra Space Use karke
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    negative = []
    positive = []

    # Negative aur positive ko alag karo
    for num in arr:
        if num < 0:
            negative.append(num)
        else:
            positive.append(num)

    # Dono ko combine karo
    return negative + positive

# Test Cases
if __name__ == "__main__":
    # Test 1
    arr1 = [1, -2, 3, -4, -1, 4, 5, -6]
    print("Original Array:", arr1)
    print("After moving negatives:", move_negatives_to_start(arr1.copy()))

    # Test 2
    arr2 = [5, -3, 2, -8, 1, -9, 7]
    print("\nOriginal Array:", arr2)
    print("Method 2:", move_negatives_method2(arr2.copy()))

    # Test 3
    arr3 = [-1, -2, -3, 4, 5, 6]
    print("\nOriginal Array:", arr3)
    print("Method 3:", move_negatives_method3(arr3.copy()))

    # Test 4 - All positive
    arr4 = [1, 2, 3, 4, 5]
    print("\nAll Positive:", arr4)
    print("Result:", move_negatives_to_start(arr4.copy()))

    # Test 5 - All negative
    arr5 = [-1, -2, -3, -4, -5]
    print("\nAll Negative:", arr5)
    print("Result:", move_negatives_to_start(arr5.copy()))

