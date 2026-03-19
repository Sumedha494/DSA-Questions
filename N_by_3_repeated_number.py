#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def majorityElement_N3(nums):
    """
    Find elements appearing more than n/3 times
    Boyer-Moore Voting Algorithm (Extended)

    Time Complexity: O(n)
    Space Complexity: O(1)

    Note: Maximum 2 elements can appear > n/3 times
    """
    if not nums:
        return []

    # Two candidates and their counts
    candidate1, candidate2 = None, None
    count1, count2 = 0, 0

    # Phase 1: Find two candidates
    for num in nums:
        if candidate1 == num:
            count1 += 1
        elif candidate2 == num:
            count2 += 1
        elif count1 == 0:
            candidate1 = num
            count1 = 1
        elif count2 == 0:
            candidate2 = num
            count2 = 1
        else:
            count1 -= 1
            count2 -= 1

    # Phase 2: Verify candidates
    result = []
    threshold = len(nums) // 3

    if nums.count(candidate1) > threshold:
        result.append(candidate1)
    if candidate2 != candidate1 and nums.count(candidate2) > threshold:
        result.append(candidate2)

    return result


def majorityElement_N3_detailed(nums):
    """
    Step-by-step explanation
    """
    print("Finding Elements > n/3 times")
    print("=" * 60)
    print("Array:", nums)
    print("n =", len(nums), ", n/3 =", len(nums) // 3)
    print("Need count >", len(nums) // 3)
    print()

    if not nums:
        return []

    c1 = c2 = None
    cnt1 = cnt2 = 0

    print("Phase 1: Finding Candidates")
    print("-" * 60)

    for i, num in enumerate(nums):
        print("Step", i + 1, ": num =", num, end=" → ")

        if c1 == num:
            cnt1 += 1
            print("c1 match, cnt1++")
        elif c2 == num:
            cnt2 += 1
            print("c2 match, cnt2++")
        elif cnt1 == 0:
            c1, cnt1 = num, 1
            print("cnt1=0, new c1 =", num)
        elif cnt2 == 0:
            c2, cnt2 = num, 1
            print("cnt2=0, new c2 =", num)
        else:
            cnt1 -= 1
            cnt2 -= 1
            print("cnt1--, cnt2--")

        print("  State: c1=", c1, "(", cnt1, "), c2=", c2, "(", cnt2, ")")

    print()
    print("Candidates found: c1 =", c1, ", c2 =", c2)

    # Phase 2: Verify
    print()
    print("Phase 2: Verification")
    print("-" * 60)

    result = []
    threshold = len(nums) // 3

    count_c1 = nums.count(c1) if c1 is not None else 0
    count_c2 = nums.count(c2) if c2 is not None else 0

    print("c1 =", c1, "appears", count_c1, "times", end=" ")
    if count_c1 > threshold:
        result.append(c1)
        print("✓ Valid!")
    else:
        print("✗ Not enough")

    if c2 != c1:
        print("c2 =", c2, "appears", count_c2, "times", end=" ")
        if count_c2 > threshold:
            result.append(c2)
            print("✓ Valid!")
        else:
            print("✗ Not enough")

    print()
    print("=" * 60)
    print("Result:", result)

    return result


def majorityElement_N3_hashmap(nums):
    """
    HashMap approach (simple but O(n) space)
    """
    from collections import Counter

    threshold = len(nums) // 3
    count = Counter(nums)

    return [num for num, cnt in count.items() if cnt > threshold]


def majorityElement_N3_sorting(nums):
    """
    Sorting approach
    Time: O(n log n)
    """
    if not nums:
        return []

    nums_sorted = sorted(nums)
    result = []
    threshold = len(nums) // 3

    i = 0
    while i < len(nums_sorted):
        count = 1
        while i + count < len(nums_sorted) and nums_sorted[i] == nums_sorted[i + count]:
            count += 1

        if count > threshold:
            result.append(nums_sorted[i])

        i += count

    return result


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ([3, 2, 3], [3]),
        ([1], [1]),
        ([1, 2], [1, 2]),
        ([1, 1, 1, 2, 2, 2], [1, 2]),
        ([1, 2, 3], []),
        ([1, 1, 1, 3, 3, 2, 2, 2], [1, 2]),
    ]

    for nums, expected in test_cases:
        result = majorityElement_N3(nums)
        status = "✅" if sorted(result) == sorted(expected) else "❌"
        print(status, nums, "→", result)

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)
    majorityElement_N3_detailed([1, 1, 1, 3, 3, 2, 2, 2])

    print("\n" + "=" * 60)
    print("TEST 3: Edge Cases")
    print("=" * 60)

    edge_cases = [
        ([], "Empty"),
        ([1], "Single"),
        ([1, 1], "Two same"),
        ([1, 2], "Two different"),
        ([1, 1, 1], "All same"),
    ]

    for nums, desc in edge_cases:
        result = majorityElement_N3(nums)
        print(desc.ljust(15), ":", nums, "→", result)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Boyer-Moore Voting (Extended for n/3)

Key Insight: Max 2 elements can appear > n/3 times

Phase 1: Find 2 candidates
Phase 2: Verify both

Time: O(n)
Space: O(1)

LeetCode #229 ✓
    """)

