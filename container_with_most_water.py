#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def maxArea(height):
    """
    Two Pointer Approach - Most Efficient
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left = 0
    right = len(height) - 1
    max_area = 0

    while left < right:
        # Width calculate karo
        width = right - left

        # Height minimum wali line ki hogi
        current_height = min(height[left], height[right])

        # Current area calculate karo
        current_area = width * current_height

        # Maximum area update karo
        max_area = max(max_area, current_area)

        # Chhoti line ko move karo (greedy approach)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area


def maxArea_bruteforce(height):
    """
    Brute Force Approach (Sabhi combinations check karo)
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    max_area = 0
    n = len(height)

    for i in range(n):
        for j in range(i + 1, n):
            width = j - i
            current_height = min(height[i], height[j])
            current_area = width * current_height
            max_area = max(max_area, current_area)

    return max_area


def maxArea_detailed(height):
    """
    Detailed explanation ke saath
    """
    left = 0
    right = len(height) - 1
    max_area = 0

    print("Step-by-step calculation:")
    step = 1

    while left < right:
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height

        print(f"Step {step}: left={left}(h={height[left]}), right={right}(h={height[right]})")
        print(f"  Width={width}, Height={current_height}, Area={current_area}")

        if current_area > max_area:
            max_area = current_area
            print(f"  ✓ New max area found: {max_area}")

        # Chhoti height wali line ko move karo
        if height[left] < height[right]:
            left += 1
            print(f"  Moving left pointer →")
        else:
            right -= 1
            print(f"  Moving right pointer ←")

        step += 1
        print()

    return max_area


# Visualization helper
def visualize_container(height, left, right):
    """Container ko visualize karne ke liye"""
    max_h = max(height)

    for level in range(max_h, 0, -1):
        line = ""
        for i, h in enumerate(height):
            if i == left or i == right:
                line += "║" if h >= level else " "
            else:
                line += "│" if h >= level else " "
            line += " "
        print(line)

    # Base line
    print("═" * (len(height) * 2))
    print(f"Left: {left}, Right: {right}")
    print(f"Area: {(right - left) * min(height[left], height[right])}\n")


# Test Cases
if __name__ == "__main__":

    # Test 1: LeetCode Example
    print("=" * 50)
    print("Test 1: [1,8,6,2,5,4,8,3,7]")
    print("=" * 50)
    height1 = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    result1 = maxArea(height1)
    print(f"Maximum Area: {result1}")
    print(f"Expected: 49\n")

    # Test 2: Simple case
    print("=" * 50)
    print("Test 2: [1,1]")
    print("=" * 50)
    height2 = [1, 1]
    result2 = maxArea(height2)
    print(f"Maximum Area: {result2}")
    print(f"Expected: 1\n")

    # Test 3: Increasing heights
    print("=" * 50)
    print("Test 3: [1,2,3,4,5]")
    print("=" * 50)
    height3 = [1, 2, 3, 4, 5]
    result3 = maxArea(height3)
    print(f"Maximum Area: {result3}\n")

    # Test 4: With detailed explanation
    print("=" * 50)
    print("Test 4: Detailed Explanation [4,3,2,1,4]")
    print("=" * 50)
    height4 = [4, 3, 2, 1, 4]
    result4 = maxArea_detailed(height4)
    print(f"Final Maximum Area: {result4}\n")

    # Test 5: Compare brute force vs optimized
    print("=" * 50)
    print("Test 5: Comparison")
    print("=" * 50)
    height5 = [1, 8, 6, 2, 5, 4, 8, 3, 7]

    import time

    start = time.time()
    result_optimized = maxArea(height5)
    time_optimized = time.time() - start

    start = time.time()
    result_brute = maxArea_bruteforce(height5)
    time_brute = time.time() - start

    print(f"Optimized Result: {result_optimized} (Time: {time_optimized:.6f}s)")
    print(f"Brute Force Result: {result_brute} (Time: {time_brute:.6f}s)")

    # Visualization example
    print("\n" + "=" * 50)
    print("Visual Representation:")
    print("=" * 50)
    height6 = [4, 3, 2, 1, 4]
    visualize_container(height6, 0, 4)
    print(f"Max Area with these pointers: {maxArea(height6)}")

