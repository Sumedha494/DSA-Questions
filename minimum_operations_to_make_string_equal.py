#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def minOperations(s1, s2):
    """
    Minimum operations to make s1 equal to s2
    Operations: Insert, Delete, Replace

    Edit Distance / Levenshtein Distance
    Time: O(m * n), Space: O(m * n)
    """
    m, n = len(s1), len(s2)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all from s1

    for j in range(n + 1):
        dp[0][j] = j  # Insert all to s1

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )

    return dp[m][n]


def minOperations_detailed(s1, s2):
    """
    Step-by-step explanation
    """
    print("Minimum Operations (Edit Distance)")
    print("=" * 60)
    print("String 1:", s1)
    print("String 2:", s2)
    print()

    m, n = len(s1), len(s2)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    print("DP Table (Initial):")
    printDP(dp, s1, s2)

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    print("\nDP Table (Final):")
    printDP(dp, s1, s2)

    print("\nMinimum Operations:", dp[m][n])

    return dp[m][n]


def printDP(dp, s1, s2):
    """
    Print DP table nicely
    """
    print("      ", end="")
    print("  -  ", end="")
    for c in s2:
        print("  " + c + "  ", end="")
    print()

    for i in range(len(dp)):
        if i == 0:
            print("  -", end=" ")
        else:
            print("  " + s1[i - 1], end=" ")

        for j in range(len(dp[0])):
            print(str(dp[i][j]).rjust(3), end="  ")
        print()


def minOperations_with_steps(s1, s2):
    """
    Return minimum operations and the actual steps
    """
    m, n = len(s1), len(s2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    # Backtrack to find operations
    operations = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            operations.append("Replace '" + s1[i - 1] + "' with '" + s2[j - 1] + "'")
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            operations.append("Insert '" + s2[j - 1] + "'")
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append("Delete '" + s1[i - 1] + "'")
            i -= 1

    operations.reverse()
    return dp[m][n], operations


def minOperations_space_optimized(s1, s2):
    """
    Space optimized: O(n) space
    """
    m, n = len(s1), len(s2)

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i

        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])

        prev, curr = curr, prev

    return prev[n]


def minOperations_only_insert_delete(s1, s2):
    """
    Only Insert and Delete allowed (no replace)
    Answer: m + n - 2 * LCS
    """
    m, n = len(s1), len(s2)

    # Find LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    # Operations = deletions + insertions
    deletions = m - lcs_length
    insertions = n - lcs_length

    return deletions + insertions


def minOperations_01_flip(s):
    """
    Minimum flips to make string alternating (0101... or 1010...)
    LeetCode #1888 style
    """
    n = len(s)

    # Cost for pattern starting with 0: 0101...
    cost_0 = 0
    # Cost for pattern starting with 1: 1010...
    cost_1 = 0

    for i in range(n):
        if i % 2 == 0:
            if s[i] != '0':
                cost_0 += 1
            if s[i] != '1':
                cost_1 += 1
        else:
            if s[i] != '1':
                cost_0 += 1
            if s[i] != '0':
                cost_1 += 1

    return min(cost_0, cost_1)


def minOperations_make_equal_01(s):
    """
    Minimum operations to make all 0s or all 1s
    """
    count_0 = s.count('0')
    count_1 = s.count('1')

    return min(count_0, count_1)


def visualize_operations(s1, s2):
    """
    Visual step-by-step transformation
    """
    print("Transformation Visualization")
    print("=" * 60)
    print("From:", s1)
    print("To:  ", s2)
    print()

    ops, steps = minOperations_with_steps(s1, s2)

    print("Operations needed:", ops)
    print()

    if steps:
        print("Steps:")
        for i, step in enumerate(steps, 1):
            print(" ", i, ".", step)
    else:
        print("Strings are already equal!")

    return ops


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ("cat", "cut", 1),
        ("abc", "abc", 0),
        ("horse", "ros", 3),
        ("intention", "execution", 5),
        ("", "abc", 3),
    ]

    for s1, s2, expected in test_cases:
        result = minOperations(s1, s2)
        status = "✅" if result == expected else "❌"
        print(status, "'" + s1 + "'", "→", "'" + s2 + "'", ":", result)

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    minOperations_detailed("cat", "cut")

    print("\n" + "=" * 60)
    print("TEST 3: With Steps")
    print("=" * 60)

    s1, s2 = "horse", "ros"
    ops, steps = minOperations_with_steps(s1, s2)

    print("From:", s1)
    print("To:  ", s2)
    print("Operations:", ops)
    print()
    print("Steps:")
    for i, step in enumerate(steps, 1):
        print(" ", i, ".", step)

    print("\n" + "=" * 60)
    print("TEST 4: Visualization")
    print("=" * 60)

    visualize_operations("abc", "axc")

    print("\n" + "=" * 60)
    print("TEST 5: Only Insert/Delete")
    print("=" * 60)

    s1, s2 = "sea", "eat"
    result = minOperations_only_insert_delete(s1, s2)
    print("From:", s1)
    print("To:  ", s2)
    print("Operations (only insert/delete):", result)
    print("Delete 's', Insert 't'")

    print("\n" + "=" * 60)
    print("TEST 6: Binary String - Alternating")
    print("=" * 60)

    binary_tests = [
        "0100",
        "1111",
        "010",
        "10",
    ]

    for s in binary_tests:
        result = minOperations_01_flip(s)
        print("'" + s + "'", "→ Alternating:", result, "flips")

    print("\n" + "=" * 60)
    print("TEST 7: Make All Same")
    print("=" * 60)

    for s in ["0100", "1111", "0000", "1010"]:
        result = minOperations_make_equal_01(s)
        print("'" + s + "'", "→ All same:", result, "flips")

    print("\n" + "=" * 60)
    print("TEST 8: Edge Cases")
    print("=" * 60)

    edge_cases = [
        ("", "", 0, "Both empty"),
        ("a", "", 1, "One empty"),
        ("", "b", 1, "Other empty"),
        ("a", "a", 0, "Same single"),
        ("a", "b", 1, "Different single"),
    ]

    for s1, s2, expected, desc in edge_cases:
        result = minOperations(s1, s2)
        status = "✅" if result == expected else "❌"
        print(status, desc.ljust(20), ":", result)

    print("\n" + "=" * 60)
    print("TEST 9: Space Optimized")
    print("=" * 60)

    s1, s2 = "intention", "execution"

    result_normal = minOperations(s1, s2)
    result_optimized = minOperations_space_optimized(s1, s2)

    print("Normal:    ", result_normal)
    print("Optimized: ", result_optimized)
    print("Match:", result_normal == result_optimized)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Minimum Operations (Edit Distance)

Problem: Transform s1 to s2 using min operations
Operations: Insert, Delete, Replace

DP Approach:
  dp[i][j] = min operations for s1[0..i-1] to s2[0..j-1]

  if s1[i-1] == s2[j-1]:
      dp[i][j] = dp[i-1][j-1]  (no operation)
  else:
      dp[i][j] = 1 + min(
          dp[i-1][j],    # Delete from s1
          dp[i][j-1],    # Insert into s1
          dp[i-1][j-1]   # Replace
      )

Time: O(m * n)
Space: O(m * n), can be O(n) with optimization

LeetCode #72: Edit Distance
    """)

    print("\n" + "=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)
    print("""
s1 = "cat", s2 = "cut"

DP Table:
        -   c   u   t
    -   0   1   2   3
    c   1   0   1   2
    a   2   1   1   2
    t   3   2   2   1  ← Answer

Operations:
  "cat" → "cut"
  Replace 'a' with 'u'

  Answer: 1 operation
    """)

    print("\n" + "=" * 60)
    print("VARIATIONS")
    print("=" * 60)
    print("""
1. Edit Distance (Insert, Delete, Replace)
   → Standard DP

2. Only Insert & Delete (No Replace)
   → Answer = m + n - 2 * LCS

3. Make Binary Alternating
   → Count mismatches for both patterns

4. Make All Same (0s or 1s)
   → min(count_0, count_1)

5. Minimum Steps to Anagram
   → Count character differences
    """)

