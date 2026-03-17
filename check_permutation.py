#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def isPermutation(s1, s2):
    """
    Check if s1 is a permutation of s2

    Approach 1: Sorting
    Time Complexity: O(n log n)
    Space Complexity: O(1) if in-place sort, O(n) for sorted()
    """
    if len(s1) != len(s2):
        return False

    return sorted(s1) == sorted(s2)


def isPermutation_hashmap(s1, s2):
    """
    Approach 2: Character Count using HashMap
    Time Complexity: O(n)
    Space Complexity: O(n) - for dictionary
    """
    if len(s1) != len(s2):
        return False

    # Count characters in s1
    char_count = {}
    for char in s1:
        char_count[char] = char_count.get(char, 0) + 1

    # Decrease count for s2
    for char in s2:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] < 0:
            return False

    # All counts should be zero
    return all(count == 0 for count in char_count.values())


def isPermutation_counter(s1, s2):
    """
    Approach 3: Using Counter from collections
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    from collections import Counter

    if len(s1) != len(s2):
        return False

    return Counter(s1) == Counter(s2)


def isPermutation_array(s1, s2):
    """
    Approach 4: Character count array (for ASCII)
    Time Complexity: O(n)
    Space Complexity: O(1) - fixed size array

    Best for ASCII or limited character set
    """
    if len(s1) != len(s2):
        return False

    # Assuming ASCII (128 characters)
    char_count = [0] * 128

    # Count characters in s1
    for char in s1:
        char_count[ord(char)] += 1

    # Decrease count for s2
    for char in s2:
        char_count[ord(char)] -= 1
        if char_count[ord(char)] < 0:
            return False

    return True


def isPermutation_xor(s1, s2):
    """
    Approach 5: XOR approach (only for checking if permutation exists)
    Works only when all characters appear even number of times

    Note: This has limitations and doesn't work for all cases
    """
    if len(s1) != len(s2):
        return False

    xor_result = 0

    for char in s1:
        xor_result ^= ord(char)

    for char in s2:
        xor_result ^= ord(char)

    return xor_result == 0


def isPermutation_detailed(s1, s2):
    """
    Step-by-step explanation
    """
    print("Checking if Permutation")
    print("=" * 70)
    print("String 1:", s1)
    print("String 2:", s2)
    print()

    # Length check
    print("Step 1: Length Check")
    print("  len(s1) =", len(s1))
    print("  len(s2) =", len(s2))

    if len(s1) != len(s2):
        print("  ❌ Different lengths! Not a permutation")
        return False

    print("  ✓ Same length, proceeding...")
    print()

    # Character counting
    print("Step 2: Count Characters in s1")
    char_count = {}
    for char in s1:
        char_count[char] = char_count.get(char, 0) + 1

    print("  Character counts:", char_count)
    print()

    # Verify with s2
    print("Step 3: Verify with s2")
    for i, char in enumerate(s2):
        print("  Position", i, ": char =", char, end=" ")

        if char not in char_count:
            print("❌ Not found in s1!")
            return False

        char_count[char] -= 1
        print("→ count =", char_count[char], end=" ")

        if char_count[char] < 0:
            print("❌ Too many occurrences!")
            return False

        print("✓")

    print()
    print("Step 4: Final Check")
    print("  All counts:", char_count)

    if all(count == 0 for count in char_count.values()):
        print("  ✅ All counts are zero! s1 is a permutation of s2")
        return True
    else:
        print("  ❌ Some counts non-zero! Not a permutation")
        return False


def checkPermutation_substring(s, pattern):
    """
    Check if any permutation of pattern exists as substring in s
    Sliding window approach

    Time Complexity: O(n)
    Space Complexity: O(1) - fixed size arrays
    """
    if len(pattern) > len(s):
        return False

    from collections import Counter

    pattern_count = Counter(pattern)
    window_count = Counter(s[:len(pattern)])

    # Check first window
    if window_count == pattern_count:
        return True

    # Slide the window
    for i in range(len(pattern), len(s)):
        # Add new character
        window_count[s[i]] += 1

        # Remove old character
        old_char = s[i - len(pattern)]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]

        # Check if current window matches
        if window_count == pattern_count:
            return True

    return False


def findAllPermutations(s, pattern):
    """
    Find all starting indices of permutations of pattern in s
    LeetCode #438 style

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    from collections import Counter

    result = []

    if len(pattern) > len(s):
        return result

    pattern_count = Counter(pattern)
    window_count = Counter()

    # Build first window
    for i in range(len(pattern)):
        window_count[s[i]] += 1

    # Check first window
    if window_count == pattern_count:
        result.append(0)

    # Slide window
    for i in range(len(pattern), len(s)):
        # Add new character
        window_count[s[i]] += 1

        # Remove old character
        old_char = s[i - len(pattern)]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]

        # Check match
        if window_count == pattern_count:
            result.append(i - len(pattern) + 1)

    return result


def visualize_permutation_check(s1, s2):
    """
    Visual representation
    """
    print("Permutation Check Visualization")
    print("=" * 70)
    print("s1:", s1)
    print("s2:", s2)
    print()

    if len(s1) != len(s2):
        print("❌ Different lengths!")
        return False

    # Show character by character
    from collections import Counter
    count1 = Counter(s1)
    count2 = Counter(s2)

    print("Character Frequency Comparison:")
    print("-" * 70)

    all_chars = sorted(set(s1) | set(s2))

    print("Char | Count in s1 | Count in s2 | Match")
    print("-" * 70)

    match = True
    for char in all_chars:
        c1 = count1.get(char, 0)
        c2 = count2.get(char, 0)
        status = "✓" if c1 == c2 else "✗"

        if c1 != c2:
            match = False

        print(repr(char).ljust(4), "|", 
              str(c1).rjust(11), "|", 
              str(c2).rjust(11), "|", 
              status)

    print("-" * 70)

    if match:
        print("✅ All characters match! Is a permutation")
    else:
        print("❌ Character counts don't match! Not a permutation")

    return match


def compare_approaches(s1, s2):
    """
    Compare different approaches
    """
    import time

    print("Performance Comparison")
    print("=" * 70)
    print("String lengths:", len(s1), "and", len(s2))
    print()

    approaches = [
        ("Sorting", isPermutation),
        ("HashMap", isPermutation_hashmap),
        ("Counter", isPermutation_counter),
        ("Array", isPermutation_array),
    ]

    for name, func in approaches:
        start = time.time()
        result = func(s1, s2)
        elapsed = time.time() - start

        print(name.ljust(15), ":", round(elapsed, 8), "s →", result)


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEST 1: Basic Permutation Check")
    print("=" * 70)

    test_cases = [
        ("abc", "bca", True),
        ("abc", "bad", False),
        ("listen", "silent", True),
        ("hello", "world", False),
        ("aab", "aba", True),
        ("aab", "aaa", False),
    ]

    for s1, s2, expected in test_cases:
        result = isPermutation(s1, s2)
        status = "✅" if result == expected else "❌"
        print(status, "s1=" + repr(s1), "s2=" + repr(s2), "→", result)

    print("\n" + "=" * 70)
    print("TEST 2: Detailed Explanation")
    print("=" * 70)
    isPermutation_detailed("abc", "bca")

    print("\n" + "=" * 70)
    print("TEST 3: Different Approaches")
    print("=" * 70)
    s1, s2 = "cinema", "iceman"
    print("Checking:", s1, "and", s2)
    print()
    print("Sorting approach:  ", isPermutation(s1, s2))
    print("HashMap approach:  ", isPermutation_hashmap(s1, s2))
    print("Counter approach:  ", isPermutation_counter(s1, s2))
    print("Array approach:    ", isPermutation_array(s1, s2))

    print("\n" + "=" * 70)
    print("TEST 4: Edge Cases")
    print("=" * 70)

    edge_cases = [
        ("", "", True, "Both empty"),
        ("a", "", False, "One empty"),
        ("a", "a", True, "Single character"),
        ("aa", "aa", True, "Repeated characters"),
        ("ab", "ba", True, "Two characters"),
        ("abc", "ab", False, "Different lengths"),
        ("  ", "  ", True, "Spaces"),
        ("a b", "b a", True, "With spaces"),
    ]

    for s1, s2, expected, desc in edge_cases:
        result = isPermutation(s1, s2)
        status = "✅" if result == expected else "❌"
        print(status, desc.ljust(25), ":", repr(s1), "vs", repr(s2), "→", result)

    print("\n" + "=" * 70)
    print("TEST 5: Permutation in Substring")
    print("=" * 70)

    s = "eidbaooo"
    pattern = "ab"
    print("String: ", s)
    print("Pattern:", pattern)
    result = checkPermutation_substring(s, pattern)
    print("Contains permutation of pattern:", result)
    print("Expected: True (ab at index 4)")

    print("\n" + "=" * 70)
    print("TEST 6: Find All Permutation Indices")
    print("=" * 70)

    s = "cbaebabacd"
    pattern = "abc"
    print("String: ", s)
    print("Pattern:", pattern)
    indices = findAllPermutations(s, pattern)
    print("Permutation found at indices:", indices)
    print("Expected: [0, 6] (cba and bac)")

    print("\n" + "=" * 70)
    print("TEST 7: Visualization")
    print("=" * 70)
    visualize_permutation_check("triangle", "integral")

    print("\n" + "=" * 70)
    print("TEST 8: Case Sensitivity")
    print("=" * 70)

    case_tests = [
        ("ABC", "abc", False, "Case sensitive"),
        ("ABC", "ABC", True, "Same case"),
        ("Abc", "cAb", True, "Mixed case"),
    ]

    for s1, s2, expected, desc in case_tests:
        result = isPermutation(s1, s2)
        status = "✅" if result == expected else "❌"
        print(status, desc.ljust(20), ":", s1, "vs", s2, "→", result)

    print("\n" + "=" * 70)
    print("TEST 9: Unicode Characters")
    print("=" * 70)

    unicode_tests = [
        ("café", "éfac", True),
        ("hello", "hëllo", False),
        ("😀😁", "😁😀", True),
    ]

    for s1, s2, expected in unicode_tests:
        result = isPermutation(s1, s2)
        status = "✅" if result == expected else "❌"
        print(status, s1, "vs", s2, "→", result)

    print("\n" + "=" * 70)
    print("TEST 10: Performance Comparison")
    print("=" * 70)

    # Create large strings
    import random
    import string

    chars = string.ascii_lowercase
    large_s1 = ''.join(random.choice(chars) for _ in range(10000))
    large_s2 = ''.join(random.sample(large_s1, len(large_s1)))

    compare_approaches(large_s1, large_s2)

    print("\n" + "=" * 70)
    print("TEST 11: LeetCode Style Problems")
    print("=" * 70)

    # Problem 1: Check if permutation (simple)
    print("Problem 1: Check Permutation")
    s1, s2 = "ab", "ba"
    print("  Input:", s1, s2)
    print("  Output:", isPermutation(s1, s2))

    # Problem 2: Permutation in string (LeetCode #567)
    print("\nProblem 2: Permutation in String (LeetCode #567)")
    s = "eidbaooo"
    pattern = "ab"
    print("  s1:", pattern)
    print("  s2:", s)
    print("  Output:", checkPermutation_substring(s, pattern))

    # Problem 3: Find all anagrams (LeetCode #438)
    print("\nProblem 3: Find All Anagrams (LeetCode #438)")
    s = "cbaebabacd"
    p = "abc"
    print("  s:", s)
    print("  p:", p)
    print("  Output:", findAllPermutations(s, p))

    print("\n" + "=" * 70)
    print("ALGORITHM SUMMARY")
    print("=" * 70)
    print("""
Check Permutation - Multiple Approaches

Definition: String s1 is a permutation of s2 if:
1. Same length
2. Same character frequencies

Approaches:

1. Sorting:
   - Sort both strings and compare
   - Time: O(n log n), Space: O(1) or O(n)

2. HashMap/Counter:
   - Count character frequencies
   - Time: O(n), Space: O(n)
   - ⭐ Most common approach

3. Character Array:
   - Fixed-size array for ASCII
   - Time: O(n), Space: O(1)
   - ⭐ Best for limited character set

4. XOR:
   - Has limitations, not recommended
   - Works only for specific cases

Extensions:
✓ Permutation in substring (sliding window)
✓ Find all permutation indices
✓ Anagram grouping

LeetCode Problems:
→ #242: Valid Anagram
→ #567: Permutation in String
→ #438: Find All Anagrams in a String

Time Complexity: O(n) with Counter ⚡
Space Complexity: O(1) with array, O(n) with HashMap
    """)

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Permutation = Same characters, different order
   "abc" and "bca" are permutations ✓

2. Anagram = Permutation of letters
   "listen" and "silent" are anagrams ✓

3. Quick checks:
   - Length must be same
   - Character frequency must match

4. Choose approach based on:
   - Character set size (ASCII vs Unicode)
   - Space constraints
   - Performance requirements

5. Sliding window for substring problems
   - Maintain character count in window
   - Slide and update counts efficiently
    """)

