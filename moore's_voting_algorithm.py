#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def mooreVotingAlgorithm(arr):
    """
    Moore's Voting Algorithm - Find Majority Element
    Majority element = element that appears > n/2 times

    Time Complexity: O(n)
    Space Complexity: O(1)

    Two phases:
    1. Find candidate
    2. Verify candidate
    """
    if not arr:
        return None

    # Phase 1: Find candidate
    candidate = None
    count = 0

    for num in arr:
        if count == 0:
            candidate = num
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1

    # Phase 2: Verify candidate
    if arr.count(candidate) > len(arr) // 2:
        return candidate

    return None


def mooreVoting_detailed(arr):
    """
    Step-by-step explanation ke saath
    """
    print("Moore's Voting Algorithm - Detailed")
    print("=" * 70)
    print("Array:", arr)
    print("Goal: Find element appearing > n/2 times")
    print("=" * 70)

    if not arr:
        print("Empty array!")
        return None

    # Phase 1: Find candidate
    print("\n📍 PHASE 1: Finding Candidate")
    print("-" * 70)

    candidate = None
    count = 0

    for i, num in enumerate(arr):
        print("Step", i + 1, ": num =", num)

        if count == 0:
            candidate = num
            count = 1
            print("  → Count is 0, set candidate =", candidate)
        elif num == candidate:
            count += 1
            print("  → Same as candidate, count++ =", count)
        else:
            count -= 1
            print("  → Different from candidate, count-- =", count)

        print("  Current: candidate =", candidate, ", count =", count)
        print()

    print("Candidate found:", candidate)

    # Phase 2: Verify
    print("\n✓ PHASE 2: Verifying Candidate")
    print("-" * 70)

    actual_count = arr.count(candidate)
    required = len(arr) // 2

    print("Candidate:", candidate)
    print("Appears:", actual_count, "times")
    print("Required: >", required, "times")

    if actual_count > required:
        print("✅ Valid! This is the majority element")
        return candidate
    else:
        print("❌ Not a majority element")
        return None


def findMajorityElement_no_verification(arr):
    """
    Without verification phase
    (Only works if majority element guaranteed to exist)
    """
    candidate = None
    count = 0

    for num in arr:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)

    return candidate


def findMajorityElement_with_index(arr):
    """
    Return majority element with all its indices
    """
    if not arr:
        return None, []

    # Find candidate
    candidate = None
    count = 0

    for num in arr:
        if count == 0:
            candidate = num
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1

    # Get all indices
    indices = [i for i, num in enumerate(arr) if num == candidate]

    # Verify
    if len(indices) > len(arr) // 2:
        return candidate, indices

    return None, []


def findMajorityElements_n3(arr):
    """
    Find elements appearing > n/3 times
    Extended Boyer-Moore algorithm

    Note: Maximum 2 elements can appear > n/3 times
    """
    if not arr:
        return []

    # Phase 1: Find two candidates
    candidate1, candidate2 = None, None
    count1, count2 = 0, 0

    for num in arr:
        if candidate1 is not None and num == candidate1:
            count1 += 1
        elif candidate2 is not None and num == candidate2:
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

    # Phase 2: Verify both candidates
    result = []

    if arr.count(candidate1) > len(arr) // 3:
        result.append(candidate1)
    if candidate2 != candidate1 and arr.count(candidate2) > len(arr) // 3:
        result.append(candidate2)

    return result


def visualize_moore_voting(arr):
    """
    Visual representation of algorithm
    """
    print("Moore's Voting Algorithm - Visualization")
    print("=" * 70)
    print("Array:", arr)
    print()

    candidate = None
    count = 0

    print("Step | Element | Action           | Candidate | Count")
    print("-" * 70)

    for i, num in enumerate(arr):
        action = ""

        if count == 0:
            candidate = num
            count = 1
            action = "Set new candidate"
        elif num == candidate:
            count += 1
            action = "Vote for candidate"
        else:
            count -= 1
            action = "Vote against"

        print(str(i + 1).rjust(4), "|", 
              str(num).rjust(7), "|", 
              action.ljust(16), "|", 
              str(candidate).rjust(9), "|", 
              str(count).rjust(5))

    print("-" * 70)
    print("\nCandidate:", candidate)

    # Verify
    actual = arr.count(candidate)
    required = len(arr) // 2

    print("Verification: appears", actual, "times (need >", str(required) + ")")

    if actual > required:
        print("✅ Majority element found:", candidate)
        return candidate
    else:
        print("❌ No majority element")
        return None


def compare_approaches(arr):
    """
    Compare Moore's algorithm with brute force
    """
    import time

    print("Comparing Approaches")
    print("=" * 70)
    print("Array size:", len(arr))
    print()

    # Moore's Algorithm
    start = time.time()
    result_moore = mooreVotingAlgorithm(arr)
    time_moore = time.time() - start

    # Brute Force (count each element)
    start = time.time()
    result_brute = None
    n = len(arr)
    for num in set(arr):
        if arr.count(num) > n // 2:
            result_brute = num
            break
    time_brute = time.time() - start

    # HashMap approach
    start = time.time()
    result_hash = None
    freq = {}
    for num in arr:
        freq[num] = freq.get(num, 0) + 1
        if freq[num] > n // 2:
            result_hash = num
            break
    time_hash = time.time() - start

    print("Moore's Algorithm (O(n), O(1)):", round(time_moore, 6), "s →", result_moore)
    print("HashMap Approach (O(n), O(n)): ", round(time_hash, 6), "s →", result_hash)
    print("Brute Force (O(n²), O(1)):    ", round(time_brute, 6), "s →", result_brute)
    print()
    print("Moore's algorithm is", round(time_brute / time_moore, 2), "x faster than brute force!")


def findAllMajorityElements_nk(arr, k):
    """
    Generalized: Find elements appearing > n/k times
    Maximum (k-1) such elements can exist
    """
    if not arr or k <= 1:
        return []

    # Phase 1: Find k-1 candidates
    candidates = {}

    for num in arr:
        if num in candidates:
            candidates[num] += 1
        elif len(candidates) < k - 1:
            candidates[num] = 1
        else:
            # Decrease all counts
            candidates = {key: val - 1 for key, val in candidates.items()}
            candidates = {key: val for key, val in candidates.items() if val > 0}

    # Phase 2: Verify candidates
    result = []
    threshold = len(arr) // k

    for candidate in candidates.keys():
        if arr.count(candidate) > threshold:
            result.append(candidate)

    return result


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEST 1: Basic Moore's Voting Algorithm")
    print("=" * 70)
    arr1 = [2, 2, 1, 1, 1, 2, 2]
    print("Array:", arr1)
    result1 = mooreVotingAlgorithm(arr1)
    print("Majority Element:", result1)
    print("Expected: 2 (appears 4 times, > 7/2)")

    print("\n" + "=" * 70)
    print("TEST 2: Detailed Step-by-Step")
    print("=" * 70)
    arr2 = [3, 3, 4, 2, 4, 4, 2, 4, 4]
    mooreVoting_detailed(arr2)

    print("\n" + "=" * 70)
    print("TEST 3: No Majority Element")
    print("=" * 70)
    arr3 = [1, 2, 3, 4, 5]
    print("Array:", arr3)
    result3 = mooreVotingAlgorithm(arr3)
    print("Majority Element:", result3)
    print("Expected: None (no element appears > n/2 times)")

    print("\n" + "=" * 70)
    print("TEST 4: With Indices")
    print("=" * 70)
    arr4 = [1, 1, 2, 1, 3, 5, 1]
    print("Array:", arr4)
    element, indices = findMajorityElement_with_index(arr4)
    print("Majority Element:", element)
    print("Appears at indices:", indices)

    print("\n" + "=" * 70)
    print("TEST 5: Find Elements > n/3")
    print("=" * 70)
    arr5 = [3, 2, 3, 1, 2, 2, 3, 1, 1]
    print("Array:", arr5)
    result5 = findMajorityElements_n3(arr5)
    print("Elements appearing > n/3 times:", result5)
    print("n/3 =", len(arr5) // 3)
    for elem in result5:
        print("  ", elem, "appears", arr5.count(elem), "times")

    print("\n" + "=" * 70)
    print("TEST 6: Visualization")
    print("=" * 70)
    arr6 = [7, 7, 5, 7, 5, 1, 5, 7, 5, 5, 7, 7, 7, 7, 7]
    visualize_moore_voting(arr6)

    print("\n" + "=" * 70)
    print("TEST 7: Edge Cases")
    print("=" * 70)

    edge_cases = [
        ([1], "Single element"),
        ([1, 1], "Two same"),
        ([1, 2], "Two different"),
        ([1, 1, 1], "All same"),
        ([1, 2, 3, 1, 1], "Simple majority"),
        ([1, 2, 1, 2, 1], "Exactly > n/2"),
        ([], "Empty array"),
    ]

    for arr, desc in edge_cases:
        result = mooreVotingAlgorithm(arr)
        print(desc.ljust(20), ":", arr, "→", result)

    print("\n" + "=" * 70)
    print("TEST 8: LeetCode Problems")
    print("=" * 70)

    # LeetCode #169: Majority Element
    arr8_1 = [3, 2, 3]
    print("Problem: Majority Element (LeetCode #169)")
    print("Array:", arr8_1)
    print("Answer:", mooreVotingAlgorithm(arr8_1))

    # LeetCode #229: Majority Element II
    arr8_2 = [3, 2, 3]
    print("\nProblem: Majority Element II (LeetCode #229)")
    print("Array:", arr8_2)
    print("Answer:", findMajorityElements_n3(arr8_2))

    print("\n" + "=" * 70)
    print("TEST 9: Performance Comparison")
    print("=" * 70)
    import random
    large_arr = [random.choice([1, 2, 3, 4, 5, 5, 5, 5]) for _ in range(1000)]
    compare_approaches(large_arr)

    print("\n" + "=" * 70)
    print("TEST 10: Generalized n/k Algorithm")
    print("=" * 70)
    arr10 = [1, 1, 1, 3, 3, 2, 2, 2]
    print("Array:", arr10)
    print("n =", len(arr10))

    for k in [2, 3, 4]:
        result = findAllMajorityElements_nk(arr10, k)
        print("Elements appearing > n/" + str(k) + ":", result)
        print("  Threshold: >", len(arr10) // k)
        for elem in result:
            print("    ", elem, "appears", arr10.count(elem), "times")
        print()

    print("=" * 70)
    print("ALGORITHM SUMMARY")
    print("=" * 70)
    print("""
Moore's Voting Algorithm (Robert S. Boyer & J Strother Moore)

Purpose: Find majority element in O(n) time and O(1) space

Majority Element: Element appearing > n/2 times

Key Idea: Cancellation/Pairing
- Same element: count++
- Different element: count--
- When count = 0: change candidate

Why it works?
→ Majority element will survive the cancellation!
→ Even if all other elements pair up, majority will remain

Two Phases:
1. Find Candidate: O(n) time, O(1) space
2. Verify: O(n) time, O(1) space

Time Complexity: O(n) ⚡
Space Complexity: O(1) 💾

Extensions:
✓ Elements > n/3: LeetCode #229 (max 2 candidates)
✓ Elements > n/k: General case (max k-1 candidates)

Applications:
✓ Election results
✓ Frequent elements
✓ Dominant patterns
✓ Network packet analysis

LeetCode Problems:
→ #169: Majority Element
→ #229: Majority Element II
    """)

    print("\n" + "=" * 70)
    print("INTUITION: Why Moore's Algorithm Works")
    print("=" * 70)
    print("""
Imagine a battlefield:
- Majority element's soldiers vs all others
- Each soldier can eliminate exactly 1 enemy
- Majority will always have soldiers left!

Example: [A, A, B, A, C, A, D]
- A appears 4 times (majority)
- Others appear 3 times total

Pairing:
A vs B → cancel
A vs C → cancel  
A vs D → cancel
A (survivor!) → This is our majority

Even in worst case, majority survives!
    """)

