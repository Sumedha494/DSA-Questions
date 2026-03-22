#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def findKthCharacter(s, k):
    """
    Find Kth character in decrypted string

    Input: "ab2cd3" -> "ababcdcdcd"
    Without fully expanding the string (memory efficient)

    Time: O(n), Space: O(1)
    """
    # Calculate expanded length
    expanded_len = 0

    for char in s:
        if char.isdigit():
            expanded_len *= int(char)
        else:
            expanded_len += 1

    # Work backwards to find kth character
    for i in range(len(s) - 1, -1, -1):
        char = s[i]

        if char.isdigit():
            expanded_len //= int(char)
            k %= expanded_len

            if k == 0:
                k = expanded_len
        else:
            if k == expanded_len or k == 0:
                return char
            expanded_len -= 1

    return ""


def findKthCharacter_simple(s, k):
    """
    Simple approach: Fully expand string
    Easy to understand but uses more memory

    Time: O(expanded length), Space: O(expanded length)
    """
    decoded = ""

    for char in s:
        if char.isdigit():
            decoded = decoded * int(char)
        else:
            decoded += char

    if k <= len(decoded):
        return decoded[k - 1]

    return ""


def findKthCharacter_detailed(s, k):
    """
    Step-by-step explanation
    """
    print("Find Kth Character in Decrypted String")
    print("=" * 60)
    print("Encoded String:", s)
    print("K:", k)
    print()

    # Step 1: Calculate expanded length
    print("Step 1: Calculate Expanded Length")
    print("-" * 40)

    expanded_len = 0

    for char in s:
        if char.isdigit():
            old_len = expanded_len
            expanded_len *= int(char)
            print("  '" + char + "' (digit): " + str(old_len) + " * " + char + " = " + str(expanded_len))
        else:
            expanded_len += 1
            print("  '" + char + "' (char):  length = " + str(expanded_len))

    print("\nTotal Expanded Length:", expanded_len)

    # Step 2: Work backwards
    print("\nStep 2: Work Backwards to Find K=" + str(k))
    print("-" * 40)

    for i in range(len(s) - 1, -1, -1):
        char = s[i]
        print("\nIndex", i, ": char = '" + char + "'")
        print("  Current: expanded_len =", expanded_len, ", k =", k)

        if char.isdigit():
            expanded_len //= int(char)
            k %= expanded_len

            if k == 0:
                k = expanded_len

            print("  Digit: expanded_len / " + char + " =", expanded_len)
            print("  New k =", k)
        else:
            if k == expanded_len or k == 0:
                print("  FOUND! Character '" + char + "' at position", k)
                return char

            expanded_len -= 1
            print("  Not found, expanded_len =", expanded_len)

    return ""


def decryptString(s):
    """
    Fully decrypt the string
    """
    decoded = ""

    for char in s:
        if char.isdigit():
            decoded = decoded * int(char)
        else:
            decoded += char

    return decoded


def decryptString_detailed(s):
    """
    Show decryption process
    """
    print("Decrypting:", s)
    print("-" * 40)

    decoded = ""

    for char in s:
        if char.isdigit():
            decoded = decoded * int(char)
            print("'" + char + "': Repeat " + char + " times -> '" + decoded + "'")
        else:
            decoded += char
            print("'" + char + "': Add char -> '" + decoded + "'")

    print("\nDecrypted:", decoded)
    return decoded


def findKthCharacter_leetcode(s, k):
    """
    LeetCode 880: Decoded String at Index
    Handles large k values efficiently
    """
    # Calculate size
    size = 0

    for char in s:
        if char.isdigit():
            size *= int(char)
        else:
            size += 1

    # Work backwards
    for i in range(len(s) - 1, -1, -1):
        char = s[i]
        k %= size

        if k == 0 and char.isalpha():
            return char

        if char.isdigit():
            size //= int(char)
        else:
            size -= 1

    return ""


# ============================================
# TEST CASES
# ============================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Basic Example")
    print("=" * 60)

    s1 = "ab2cd3"
    print("Encoded:", s1)

    decoded1 = decryptString(s1)
    print("Decoded:", decoded1)
    print("Length: ", len(decoded1))

    for k in [1, 2, 3, 4, 5, 6]:
        result = findKthCharacter(s1, k)
        print("K=" + str(k) + ":", result, "(expected: '" + decoded1[k-1] + "')")

    print("\n" + "=" * 60)
    print("TEST 2: Detailed Explanation")
    print("=" * 60)

    s2 = "a2b3"
    print("Encoded:", s2)
    decryptString_detailed(s2)
    print()
    findKthCharacter_detailed(s2, 5)

    print("\n" + "=" * 60)
    print("TEST 3: Different Examples")
    print("=" * 60)

    test_cases = [
        ("abc2", 4, "a"),      # abcabc -> 4th is 'a'
        ("abc2", 5, "b"),      # abcabc -> 5th is 'b'
        ("a2b2", 5, "b"),      # aabb -> but wait...
        ("leet2code3", 10, "o"),
    ]

    for s, k, expected in test_cases:
        decoded = decryptString(s)
        result = findKthCharacter(s, k)

        print("Encoded:", s)
        print("Decoded:", decoded)
        print("K=" + str(k) + ": Got '" + result + "'")
        print()

    print("=" * 60)
    print("TEST 4: Large K (Efficient)")
    print("=" * 60)

    s4 = "a2345678999999999999999"
    k4 = 1

    print("Encoded:", s4)
    print("K:", k4)
    print("Note: Cannot expand this string (too large!)")

    result4 = findKthCharacter_leetcode(s4, k4)
    print("Result:", result4)

    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)

    edge_cases = [
        ("a", 1, "Single char"),
        ("a2", 1, "First char"),
        ("a2", 2, "Repeated char"),
        ("ab2", 3, "After repeat"),
        ("abc", 2, "No digits"),
    ]

    for s, k, desc in edge_cases:
        decoded = decryptString(s)
        result = findKthCharacter(s, k)
        print(desc + ":")
        print("  s='" + s + "', k=" + str(k))
        print("  Decoded: '" + decoded + "'")
        print("  Result: '" + result + "'")
        print()

    print("=" * 60)
    print("TEST 6: Visualization")
    print("=" * 60)

    s6 = "ab2c3"
    print("Step-by-step decoding:", s6)
    print()
    print("a    -> 'a'")
    print("ab   -> 'ab'")
    print("ab2  -> 'abab'")
    print("ab2c -> 'ababc'")
    print("ab2c3-> 'ababcababcababc'")
    print()

    decoded6 = decryptString(s6)
    print("Final:", decoded6)
    print("Length:", len(decoded6))
    print()

    print("Finding each character:")
    for i in range(1, len(decoded6) + 1):
        result = findKthCharacter(s6, i)
        print("K=" + str(i).rjust(2) + ":", result)

    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    print("""
Find Kth Character in Decrypted String

Problem: "ab2cd3" -> "ababcdcdcd", find Kth char

Approach 1: Simple (Expand fully)
  - Decode entire string
  - Return k-1 index
  - Time: O(expanded), Space: O(expanded)
  - Problem: May be too large!

Approach 2: Efficient (Work backwards)
  Step 1: Calculate total expanded length
  Step 2: Work backwards through string
    - If digit: size /= digit, k %= size
    - If char: check if k == size
  - Time: O(n), Space: O(1)

Key Insight:
  "ab2" = "abab"
  Position 3 in "abab" = Position 1 in "ab"
  Because 3 % 2 = 1

LeetCode #880: Decoded String at Index
    """)

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
Example: s = "ab2", k = 3

Expanded: "abab" (length 4)
          1234

k=3 is same as k=1 (because 3 % 2 = 1)

Why? Because "ab" repeats!
Position 3 in "abab" = Position 1 in "ab" = 'a'

Working Backwards:
- See '2': size = 4 -> 2, k = 3 % 2 = 1
- See 'b': size = 2, k = 1, not equal
- See 'a': size = 1, k = 1, FOUND! Return 'a'
    """)

