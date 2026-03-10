#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def isSubsequence(s, t):
    """
    Two Pointer Approach - Most Efficient
    Time Complexity: O(n) where n = len(t)
    Space Complexity: O(1)

    Check karta hai ki s, t ka subsequence hai ya nahi
    """
    # Edge case: empty string always subsequence hai
    if not s:
        return True

    i = 0  # pointer for s
    j = 0  # pointer for t

    while j < len(t):
        if s[i] == t[j]:
            i += 1
            # Agar saare characters match ho gaye
            if i == len(s):
                return True
        j += 1

    return False


def isSubsequence_recursive(s, t, i=0, j=0):
    """
    Recursive Approach
    Time Complexity: O(n)
    Space Complexity: O(n) - recursion stack
    """
    # Base cases
    if i == len(s):
        return True
    if j == len(t):
        return False

    # Agar characters match ho gaye
    if s[i] == t[j]:
        return isSubsequence_recursive(s, t, i + 1, j + 1)
    else:
        return isSubsequence_recursive(s, t, i, j + 1)


def isSubsequence_detailed(s, t):
    """
    Detailed explanation ke saath
    """
    print(f"Checking if '{s}' is subsequence of '{t}'")
    print("-" * 50)

    if not s:
        print("Empty string is always a subsequence")
        return True

    i = 0
    j = 0
    matched = []

    while j < len(t):
        print(f"Step {j+1}: Comparing s[{i}]='{s[i]}' with t[{j}]='{t[j]}'", end=" ")

        if s[i] == t[j]:
            print(f"✓ Match! Moving both pointers")
            matched.append((i, j, s[i]))
            i += 1

            if i == len(s):
                print(f"\n✅ All characters matched!")
                print(f"Matched positions: {matched}")
                return True
        else:
            print(f"✗ No match, moving t pointer only")

        j += 1

    print(f"\n❌ Could not match all characters")
    print(f"Matched so far: {matched}")
    return False


def findAllSubsequences(t, s):
    """
    Find all possible ways s can be a subsequence of t
    """
    def backtrack(t_idx, s_idx, path):
        if s_idx == len(s):
            result.append(path[:])
            return

        if t_idx == len(t):
            return

        for i in range(t_idx, len(t)):
            if t[i] == s[s_idx]:
                path.append((i, t[i]))
                backtrack(i + 1, s_idx + 1, path)
                path.pop()

    result = []
    backtrack(0, 0, [])
    return result


def longestCommonSubsequence(s1, s2):
    """
    Bonus: Longest Common Subsequence using DP
    Time Complexity: O(m*n)
    Space Complexity: O(m*n)
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def countDistinctSubsequences(s, t):
    """
    Count kitne distinct ways mein s, t ka subsequence hai
    Time Complexity: O(m*n)
    Space Complexity: O(m*n)
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Empty string always 1 way se subsequence hai
    for i in range(m + 1):
        dp[i][0] = 1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j]  # Don't use s[i-1]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]  # Use s[i-1]

    return dp[m][n]


# Test Cases
if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Examples")
    print("=" * 60)

    test_cases = [
        ("abc", "ahbgdc", True),
        ("axc", "ahbgdc", False),
        ("", "ahbgdc", True),
        ("b", "abc", True),
        ("ace", "abcde", True),
        ("aec", "abcde", False),
    ]

    for s, t, expected in test_cases:
        result = isSubsequence(s, t)
        status = "✅" if result == expected else "❌"
        print(f"{status} s='{s}', t='{t}' → {result} (Expected: {expected})")

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)
    isSubsequence_detailed("ace", "abcde")

    print("\n" + "=" * 60)
    isSubsequence_detailed("aec", "abcde")

    print("\n" + "=" * 60)
    print("TEST 3: Recursive Approach")
    print("=" * 60)
    s, t = "abc", "ahbgdc"
    result = isSubsequence_recursive(s, t)
    print(f"Recursive: Is '{s}' subsequence of '{t}'? {result}")

    print("\n" + "=" * 60)
    print("TEST 4: Find All Subsequence Paths")
    print("=" * 60)
    s, t = "ab", "aabbb"
    paths = findAllSubsequences(t, s)
    print(f"Finding all ways '{s}' is subsequence of '{t}':")
    for idx, path in enumerate(paths, 1):
        print(f"  Way {idx}: {path}")

    print("\n" + "=" * 60)
    print("TEST 5: Count Distinct Subsequences")
    print("=" * 60)
    test_count = [
        ("rabbbit", "rabbit"),
        ("babgbag", "bag"),
    ]

    for t, s in test_count:
        count = countDistinctSubsequences(t, s)
        print(f"'{s}' appears {count} times as subsequence in '{t}'")

    print("\n" + "=" * 60)
    print("TEST 6: Longest Common Subsequence")
    print("=" * 60)
    lcs_tests = [
        ("abcde", "ace"),
        ("abc", "abc"),
        ("abc", "def"),
    ]

    for s1, s2 in lcs_tests:
        lcs_len = longestCommonSubsequence(s1, s2)
        print(f"LCS of '{s1}' and '{s2}': {lcs_len}")

    print("\n" + "=" * 60)
    print("TEST 7: Edge Cases")
    print("=" * 60)
    edge_cases = [
        ("", "", True),
        ("a", "", False),
        ("", "abc", True),
        ("aaa", "aaa", True),
        ("aaa", "aa", False),
    ]

    for s, t, expected in edge_cases:
        result = isSubsequence(s, t)
        status = "✅" if result == expected else "❌"
        print(f"{status} s='{s}', t='{t}' → {result}")

