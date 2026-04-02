#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def fourSum(nums, target):
    """
    Find all unique quadruplets that sum to target

    Two Pointer Approach
    Time: O(n³), Space: O(1) excluding output
    """
    if not nums or len(nums) < 4:
        return []

    nums.sort()
    n = len(nums)
    result = []

    for i in range(n - 3):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        for j in range(i + 1, n - 2):
            # Skip duplicates for second number
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            # Two pointer for remaining two numbers
            left = j + 1
            right = n - 1

            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]

                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])

                    # Skip duplicates
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1

                elif current_sum < target:
                    left += 1
                else:
                    right -= 1

    return result


def fourSum_detailed(nums, target):
    """
    Step-by-step explanation
    """
    print("4 Sum Problem")
    print("=" * 60)
    print("Array:", nums)
    print("Target:", target)
    print()

    if len(nums) < 4:
        print("Array too small!")
        return []

    nums.sort()
    print("Sorted:", nums)
    print()

    n = len(nums)
    result = []

    print("Finding quadruplets:")
    print("-" * 60)

    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            left = j + 1
            right = n - 1

            print("\nFixed: nums[" + str(i) + "]=" + str(nums[i]) + 
                  ", nums[" + str(j) + "]=" + str(nums[j]))
            print("Looking for sum =", target - nums[i] - nums[j])

            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]

                quad = [nums[i], nums[j], nums[left], nums[right]]

                print("  Try:", quad, "sum =", current_sum, end=" ")

                if current_sum == target:
                    print("✓ FOUND")
                    result.append(quad)

                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1

                elif current_sum < target:
                    print("(too small)")
                    left += 1
                else:
                    print("(too large)")
                    right -= 1

    print()
    print("=" * 60)
    print("All quadruplets:", result)
    print("Count:", len(result))

    return result


def fourSum_hashmap(nums, target):
    """
    Using HashMap
    Time: O(n²) average, Space: O(n²)
    """
    if len(nums) < 4:
        return []

    nums.sort()
    n = len(nums)
    result = set()

    # Store all pair sums
    pair_sums = {}

    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = nums[i] + nums[j]
            if pair_sum not in pair_sums:
                pair_sums[pair_sum] = []
            pair_sums[pair_sum].append((i, j))

    # Find complementary pairs
    for i in range(n):
        for j in range(i + 1, n):
            current_sum = nums[i] + nums[j]
            complement = target - current_sum

            if complement in pair_sums:
                for k, l in pair_sums[complement]:
                    # Ensure no index overlap
                    if k > j:
                        quad = tuple(sorted([nums[i], nums[j], nums[k], nums[l]]))
                        result.add(quad)

    return [list(quad) for quad in result]


def fourSum_kSum_generalized(nums, target, k):
    """
    Generalized K-Sum problem
    Works for 2-sum, 3-sum, 4-sum, etc.
    """
    def kSum(nums, target, k, start=0):
        result = []

        if k == 2:
            # Two pointer
            left = start
            right = len(nums) - 1

            while left < right:
                current_sum = nums[left] + nums[right]

                if current_sum == target:
                    result.append([nums[left], nums[right]])

                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1

                elif current_sum < target:
                    left += 1
                else:
                    right -= 1

        else:
            # Recursive k-sum
            for i in range(start, len(nums) - k + 1):
                if i > start and nums[i] == nums[i - 1]:
                    continue

                sub_result = kSum(nums, target - nums[i], k - 1, i + 1)

                for sub in sub_result:
                    result.append([nums[i]] + sub)

        return result

    nums.sort()
    return kSum(nums, target, k)


def fourSum_count(nums, target):
    """
    Count number of quadruplets (including duplicates)
    """
    nums.sort()
    n = len(nums)
    count = 0

    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            left = j + 1
            right = n - 1

            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]

                if current_sum == target:
                    count += 1
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1

    return count


def visualize_fourSum(nums, target):
    """
    Visual representation
    """
    print("4 Sum Visualization")
    print("=" * 60)
    print("Array:", nums)
    print("Target:", target)
    print()

    nums.sort()
    print("Sorted:", nums)
    print()

    print("Strategy: Fix 2 numbers, use two pointers for rest")
    print()

    result = []
    n = len(nums)

    for i in range(min(2, n - 3)):  # Show first 2 iterations
        for j in range(i + 1, min(i + 3, n - 2)):
            print("Fixed: nums[" + str(i) + "]=" + str(nums[i]) + 
                  ", nums[" + str(j) + "]=" + str(nums[j]))

            left = j + 1
            right = n - 1

            print("  Range: [" + str(left) + ", " + str(right) + "]")
            print("  Looking for:", target - nums[i] - nums[j])

            attempts = 0
            while left < right and attempts < 3:  # Show first 3 attempts
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                quad = [nums[i], nums[j], nums[left], nums[right]]

                print("    ", quad, "=", current_sum)

                if current_sum == target:
                    result.append(quad)
                    break
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1

                attempts += 1

            print()

    full_result = fourSum(nums, target)
    print("Total quadruplets found:", len(full_result))

    return full_result


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ([1, 0, -1, 0, -2, 2], 0),
        ([2, 2, 2, 2, 2], 8),
        ([1, 2, 3, 4, 5], 10),
    ]

    for nums, target in test_cases:
        result = fourSum(nums, target)
        print("nums:", nums)
        print("target:", target)
        print("result:", result)
        print()

    print("=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    fourSum_detailed([1, 0, -1, 0, -2, 2], 0)

    print("\n" + "=" * 60)
    print("TEST 3: Edge Cases")
    print("=" * 60)

    # Empty
    print("Empty array:", fourSum([], 0))

    # Too small
    print("Array < 4:", fourSum([1, 2, 3], 6))

    # Exact 4 elements
    print("Exact 4 elements [1,2,3,4], target=10:", 
          fourSum([1, 2, 3, 4], 10))

    # All same
    print("All same [2,2,2,2,2], target=8:", 
          fourSum([2, 2, 2, 2, 2], 8))

    # No solution
    print("No solution [1,2,3,4], target=100:", 
          fourSum([1, 2, 3, 4], 100))

    print("\n" + "=" * 60)
    print("TEST 4: Duplicates Handling")
    print("=" * 60)

    nums4 = [0, 0, 0, 0, 1, 1, 1, 1]
    target4 = 2
    result4 = fourSum(nums4, target4)

    print("Array:", nums4)
    print("Target:", target4)
    print("Quadruplets:", result4)
    print("(Should have no duplicates)")

    print("\n" + "=" * 60)
    print("TEST 5: Count Quadruplets")
    print("=" * 60)

    nums5 = [1, 0, -1, 0, -2, 2]
    target5 = 0
    count5 = fourSum_count(nums5, target5)
    unique5 = len(fourSum(nums5, target5))

    print("Array:", nums5)
    print("Target:", target5)
    print("Total quadruplets (with duplicates):", count5)
    print("Unique quadruplets:", unique5)

    print("\n" + "=" * 60)
    print("TEST 6: Generalized K-Sum")
    print("=" * 60)

    nums6 = [1, 0, -1, 0, -2, 2]
    target6 = 0

    print("Array:", nums6)
    print("Target:", target6)
    print()

    for k in [2, 3, 4]:
        result = fourSum_kSum_generalized(nums6, target6, k)
        print(str(k) + "-Sum:", result[:3], "... (", len(result), "total)")

    print("\n" + "=" * 60)
    print("TEST 7: Visualization")
    print("=" * 60)

    visualize_fourSum([1, 0, -1, 0, -2, 2], 0)

    print("\n" + "=" * 60)
    print("TEST 8: Large Target")
    print("=" * 60)

    nums8 = [1, 2, 3, 4, 5, 6, 7, 8]
    target8 = 20
    result8 = fourSum(nums8, target8)

    print("Array:", nums8)
    print("Target:", target8)
    print("Result:", result8)

    print("\n" + "=" * 60)
    print("TEST 9: Negative Numbers")
    print("=" * 60)

    nums9 = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    target9 = 0
    result9 = fourSum(nums9, target9)

    print("Array:", nums9)
    print("Target:", target9)
    print("Found", len(result9), "quadruplets")
    print("First 3:", result9[:3])

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
4 Sum Problem

Problem: Find all unique quadruplets [a, b, c, d] 
         where a + b + c + d = target

Approach: Fix two numbers, two pointer for rest

Algorithm:
  1. Sort array
  2. Fix first number (i)
  3. Fix second number (j)
  4. Use two pointers (left, right) for rest
  5. Skip duplicates

Time Complexity: O(n³)
  - Two loops: O(n²)
  - Two pointer: O(n)
  - Total: O(n³)

Space Complexity: O(1) excluding output

Similar to:
  - 2 Sum: O(n)
  - 3 Sum: O(n²)
  - 4 Sum: O(n³)
  - K Sum: O(n^(k-1))

LeetCode #18: 4Sum
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print("""
Array: [1, 0, -1, 0, -2, 2]
Target: 0

After sorting: [-2, -1, 0, 0, 1, 2]

Fix i=0 (-2), j=1 (-1):
  [-2, -1, 0, 0, 1, 2]
    i   j   L        R

  Sum = -2 + -1 + 0 + 2 = -1 (< 0)
  Move L →

  Sum = -2 + -1 + 0 + 1 = -2 (< 0)
  Move L →

  Sum = -2 + -1 + 1 + 2 = 0 ✓ FOUND!
  [-2, -1, 1, 2]

Continue for all combinations...
    """)

    print("\n" + "=" * 60)
    print("KEY PATTERNS")
    print("=" * 60)
    print("""
1. Sort array first
2. Fix k-2 numbers
3. Two pointer for last 2
4. Skip duplicates:
   - After finding solution
   - When fixing numbers

Duplicate Handling:
  if i > 0 and nums[i] == nums[i-1]:
      continue  # Skip duplicate

Time Complexity Pattern:
  2-Sum: O(n)    - Two pointer
  3-Sum: O(n²)   - Fix 1, two pointer
  4-Sum: O(n³)   - Fix 2, two pointer
  K-Sum: O(n^(k-1))
    """)

