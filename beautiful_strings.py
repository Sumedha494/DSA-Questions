#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def is_beautiful_no_adjacent(s):
    """
    Check if string has no two adjacent same characters
    Time: O(n), Space: O(1)
    """
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            return False
    return True

# Examples
test_strings = [
    "aba",
    "abab",
    "abc",
    "aab",
    "aa",
    "abba",
    "abcabc",
    "a",
    "",
]

print("✨ BEAUTIFUL STRING (No Adjacent Same)")
print("=" * 45)

for s in test_strings:
    result = is_beautiful_no_adjacent(s)
    status = "✅ Beautiful" if result else "❌ Not Beautiful"
    print(f"'{s}' → {status}")


# In[ ]:


def make_beautiful_remove(s):
    """
    Make string beautiful by removing minimum characters
    Remove adjacent duplicates
    """
    if not s:
        return "", 0

    result = [s[0]]
    removed = 0

    for i in range(1, len(s)):
        if s[i] != result[-1]:
            result.append(s[i])
        else:
            removed += 1

    return ''.join(result), removed

# Examples
test_strings = [
    "aab",
    "aaab",
    "aabbcc",
    "abba",
    "aaabbbccc",
    "abcabc",
]

print("✨ MAKE BEAUTIFUL (Remove Adjacent)")
print("=" * 55)

for s in test_strings:
    beautiful, removed = make_beautiful_remove(s)
    print(f"'{s}' → '{beautiful}' (removed {removed})")


# In[ ]:


def min_deletions_beautiful(s):
    """
    Minimum deletions to make string beautiful
    (No two adjacent same characters)

    Time: O(n), Space: O(1)
    """
    if not s:
        return 0

    deletions = 0

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            deletions += 1

    return deletions

def min_deletions_visualized(s):
    """
    Visualize the deletion process
    """
    print(f"🔍 MINIMUM DELETIONS FOR '{s}'")
    print("=" * 50)

    if not s:
        print("Empty string - already beautiful!")
        return 0

    deletions = []

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            deletions.append(i)
            print(f"   Position {i}: '{s[i]}' same as '{s[i-1]}' → Delete")

    # Build result
    result = ''.join(c for i, c in enumerate(s) if i not in deletions)

    print("-" * 50)
    print(f"📊 Deletions needed: {len(deletions)}")
    print(f"📍 Delete positions: {deletions}")
    print(f"✅ Beautiful string: '{result}'")

    return len(deletions)

# Example
s = "aabbbcca"
min_deletions_visualized(s)


# In[ ]:


def make_alternating(s, chars='ab'):
    """
    Make string alternating using only given characters
    Returns minimum changes needed
    """
    if len(chars) < 2:
        return -1, ""

    char1, char2 = chars[0], chars[1]

    # Pattern 1: starts with char1 (ababab...)
    changes1 = 0
    result1 = []

    for i, c in enumerate(s):
        expected = char1 if i % 2 == 0 else char2
        if c != expected:
            changes1 += 1
        result1.append(expected)

    # Pattern 2: starts with char2 (bababa...)
    changes2 = 0
    result2 = []

    for i, c in enumerate(s):
        expected = char2 if i % 2 == 0 else char1
        if c != expected:
            changes2 += 1
        result2.append(expected)

    if changes1 <= changes2:
        return changes1, ''.join(result1)
    else:
        return changes2, ''.join(result2)

# Examples
test_strings = [
    "aaaa",
    "abab",
    "baba",
    "aabb",
    "abba",
    "aaabbb",
]

print("✨ MAKE ALTERNATING BEAUTIFUL")
print("=" * 50)

for s in test_strings:
    changes, result = make_alternating(s)
    print(f"'{s}' → '{result}' ({changes} changes)")


# In[ ]:


def beautiful_binary(s):
    """
    Make binary string beautiful by ensuring no "010" substring
    LeetCode 1758 variant

    Returns minimum changes
    """
    changes = 0
    s_list = list(s)
    i = 0

    while i <= len(s_list) - 3:
        if s_list[i:i+3] == ['0', '1', '0']:
            # Change the third character to avoid "010"
            s_list[i + 2] = '1'
            changes += 1
            i += 3  # Skip ahead
        else:
            i += 1

    return changes, ''.join(s_list)

# Examples
test_strings = [
    "010",
    "0100",
    "01010",
    "010010",
    "111",
    "000",
]

print("✨ BEAUTIFUL BINARY STRING (No '010')")
print("=" * 50)

for s in test_strings:
    changes, result = beautiful_binary(s)
    print(f"'{s}' → '{result}' ({changes} changes)")


# In[ ]:


def smallest_beautiful_string(s, k):
    """
    Find lexicographically smallest beautiful string
    that is greater than s

    Beautiful = No palindrome of length 2 or 3
    Uses first k letters of alphabet

    LeetCode 2663
    """
    n = len(s)
    s_list = list(s)

    # Start from rightmost position
    i = n - 1

    while i >= 0:
        # Try to increment current character
        s_list[i] = chr(ord(s_list[i]) + 1)

        # Check if valid character (within first k letters)
        while s_list[i] < chr(ord('a') + k):
            # Check for palindrome
            if is_valid_position(s_list, i):
                # Fill rest with smallest valid characters
                for j in range(i + 1, n):
                    s_list[j] = 'a'
                    while not is_valid_position(s_list, j):
                        s_list[j] = chr(ord(s_list[j]) + 1)

                return ''.join(s_list)

            s_list[i] = chr(ord(s_list[i]) + 1)

        i -= 1

    return ""  # No valid string exists

def is_valid_position(s, i):
    """Check if position i doesn't create palindrome"""
    # Check length 2 palindrome
    if i >= 1 and s[i] == s[i - 1]:
        return False

    # Check length 3 palindrome
    if i >= 2 and s[i] == s[i - 2]:
        return False

    return True

# Examples
test_cases = [
    ("abcz", 26),
    ("abc", 4),
    ("ab", 3),
]

print("✨ SMALLEST BEAUTIFUL STRING")
print("=" * 45)

for s, k in test_cases:
    result = smallest_beautiful_string(s, k)
    print(f"'{s}' (k={k}) → '{result}'")


# In[ ]:


def beautiful_arrangement(s):
    """
    Rearrange string so no two same characters are adjacent
    Returns empty string if impossible
    """
    from collections import Counter
    import heapq

    # Count frequency
    freq = Counter(s)
    n = len(s)

    # Check if possible
    max_freq = max(freq.values())
    if max_freq > (n + 1) // 2:
        return "", False

    # Use max heap (negative for max heap)
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)

    result = []
    prev_count, prev_char = 0, ''

    while heap:
        count, char = heapq.heappop(heap)
        result.append(char)

        # Add previous character back if it has remaining count
        if prev_count < 0:
            heapq.heappush(heap, (prev_count, prev_char))

        # Update previous
        prev_count = count + 1  # Used one
        prev_char = char

    return ''.join(result), True

# Examples
test_strings = [
    "aab",
    "aaab",
    "aabb",
    "aaabbc",
    "aaaa",
    "aabbcc",
]

print("✨ BEAUTIFUL ARRANGEMENT")
print("=" * 50)

for s in test_strings:
    result, possible = beautiful_arrangement(s)

    if possible:
        print(f"'{s}' → '{result}' ✅")
    else:
        print(f"'{s}' → Impossible ❌")


# In[ ]:


def make_beautiful_visualized(s):
    """
    Visualize making string beautiful step by step
    """
    from collections import Counter
    import heapq

    print("✨ BEAUTIFUL STRING ARRANGEMENT")
    print("=" * 55)
    print(f"Input: '{s}'")

    freq = Counter(s)
    n = len(s)

    print(f"\n📊 Character Frequencies:")
    for char, count in sorted(freq.items()):
        bar = "█" * count
        print(f"   '{char}': {count} {bar}")

    max_freq = max(freq.values())
    threshold = (n + 1) // 2

    print(f"\n📏 Max frequency: {max_freq}")
    print(f"📏 Threshold: {threshold} (n+1)//2")

    if max_freq > threshold:
        print(f"\n❌ IMPOSSIBLE! Max freq ({max_freq}) > threshold ({threshold})")
        return ""

    # Build beautiful string
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)

    result = []
    prev_count, prev_char = 0, ''
    step = 0

    print(f"\n📊 Building Beautiful String:")
    print("-" * 50)

    while heap:
        count, char = heapq.heappop(heap)
        step += 1

        result.append(char)

        print(f"Step {step}: Pick '{char}' (remaining: {-count-1})")
        print(f"         Current: '{''.join(result)}'")

        if prev_count < 0:
            heapq.heappush(heap, (prev_count, prev_char))

        prev_count = count + 1
        prev_char = char

    beautiful = ''.join(result)

    print("-" * 50)
    print(f"\n✅ Beautiful String: '{beautiful}'")

    # Verify
    is_valid = all(beautiful[i] != beautiful[i+1] for i in range(len(beautiful)-1))
    print(f"✅ Valid: {is_valid}")

    return beautiful

# Example
s = "aabbcc"
make_beautiful_visualized(s)


# In[ ]:


def count_beautiful_substrings(s):
    """
    Count substrings where no two adjacent characters are same
    """
    n = len(s)
    count = 0
    beautiful_substrings = []

    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = s[i:j]

            # Check if beautiful
            is_beautiful = True
            for k in range(len(substring) - 1):
                if substring[k] == substring[k + 1]:
                    is_beautiful = False
                    break

            if is_beautiful:
                count += 1
                beautiful_substrings.append(substring)

    return count, beautiful_substrings

# Example
s = "abab"

count, substrings = count_beautiful_substrings(s)

print(f"✨ BEAUTIFUL SUBSTRINGS OF '{s}'")
print("=" * 45)
print(f"Count: {count}")
print(f"Substrings: {substrings}")


# In[ ]:


def is_beautiful_unique(s):
    """
    Check if string has all unique characters (beautiful)
    """
    return len(s) == len(set(s))

def make_beautiful_unique(s):
    """
    Make string have unique characters by removing duplicates
    Keep first occurrence
    """
    seen = set()
    result = []
    removed = []

    for i, char in enumerate(s):
        if char not in seen:
            seen.add(char)
            result.append(char)
        else:
            removed.append((i, char))

    return ''.join(result), removed

# Examples
test_strings = [
    "abcdef",
    "aabbcc",
    "hello",
    "programming",
    "beautiful",
]

print("✨ BEAUTIFUL STRING (Unique Characters)")
print("=" * 55)

for s in test_strings:
    is_beautiful = is_beautiful_unique(s)
    beautiful, removed = make_beautiful_unique(s)

    status = "✅" if is_beautiful else "❌"

    print(f"'{s}'")
    print(f"   Original: {status} {'Beautiful' if is_beautiful else 'Not Beautiful'}")
    print(f"   Made Beautiful: '{beautiful}'")
    print(f"   Removed: {len(removed)} chars at {[r[0] for r in removed]}")
    print()


# In[ ]:


def beautiful_string_tool():
    """Interactive beautiful string tool"""

    print("✨ BEAUTIFUL STRING TOOL")
    print("=" * 50)

    s = input("Enter string: ")

    print("\n📋 SELECT OPERATION:")
    print("1. Check if beautiful (no adjacent same)")
    print("2. Make beautiful (remove adjacent)")
    print("3. Make alternating")
    print("4. Rearrange beautifully")
    print("5. Count beautiful substrings")
    print("6. Full analysis")

    choice = input("\nChoice (1-6): ")

    print("\n" + "=" * 50)
    print(f"Input: '{s}'")
    print("-" * 50)

    if choice == '1':
        result = all(s[i] != s[i+1] for i in range(len(s)-1)) if len(s) > 1 else True
        print(f"Beautiful: {'✅ Yes' if result else '❌ No'}")

    elif choice == '2':
        result = []
        for c in s:
            if not result or c != result[-1]:
                result.append(c)
        print(f"Result: '{''.join(result)}'")
        print(f"Removed: {len(s) - len(result)} characters")

    elif choice == '3':
        changes, result = make_alternating(s)
        print(f"Alternating: '{result}'")
        print(f"Changes: {changes}")

    elif choice == '4':
        result, possible = beautiful_arrangement(s)
        if possible:
            print(f"Rearranged: '{result}'")
        else:
            print("❌ Impossible to rearrange beautifully!")

    elif choice == '5':
        count, substrings = count_beautiful_substrings(s)
        print(f"Beautiful substrings: {count}")
        print(f"Examples: {substrings[:10]}{'...' if len(substrings) > 10 else ''}")

    elif choice == '6':
        # Full analysis
        is_beautiful = all(s[i] != s[i+1] for i in range(len(s)-1)) if len(s) > 1 else True

        # Make beautiful by removing
        removed_result = []
        for c in s:
            if not removed_result or c != removed_result[-1]:
                removed_result.append(c)

        # Rearrange
        rearranged, possible = beautiful_arrangement(s)

        print(f"\n📊 FULL ANALYSIS:")
        print(f"   Is Beautiful: {'✅' if is_beautiful else '❌'}")
        print(f"   By Removing: '{''.join(removed_result)}' ({len(s)-len(removed_result)} removed)")
        print(f"   By Rearranging: {f\"'{rearranged}'\" if possible else 'Impossible'}")

    print("=" * 50)

# Run
beautiful_string_tool()


# In[ ]:


from collections import Counter
import heapq

class BeautifulString:
    """Complete utility for beautiful string operations"""

    def __init__(self, s):
        self.s = s
        self.n = len(s)
        self.freq = Counter(s)

    def is_beautiful_no_adjacent(self):
        """No two adjacent same characters"""
        for i in range(self.n - 1):
            if self.s[i] == self.s[i + 1]:
                return False
        return True

    def is_beautiful_unique(self):
        """All unique characters"""
        return self.n == len(set(self.s))

    def is_beautiful_alternating(self, chars='ab'):
        """Alternating pattern"""
        if self.n == 0:
            return True

        for i in range(self.n):
            expected = chars[i % 2]
            if self.s[i] != expected:
                return False
        return True

    def min_deletions_no_adjacent(self):
        """Minimum deletions for no adjacent same"""
        deletions = 0
        for i in range(1, self.n):
            if self.s[i] == self.s[i - 1]:
                deletions += 1
        return deletions

    def make_beautiful_remove(self):
        """Remove to make beautiful"""
        if not self.s:
            return ""

        result = [self.s[0]]
        for i in range(1, self.n):
            if self.s[i] != result[-1]:
                result.append(self.s[i])

        return ''.join(result)

    def can_rearrange_beautiful(self):
        """Check if can rearrange beautifully"""
        max_freq = max(self.freq.values()) if self.freq else 0
        return max_freq <= (self.n + 1) // 2

    def rearrange_beautiful(self):
        """Rearrange to make beautiful"""
        if not self.can_rearrange_beautiful():
            return None

        heap = [(-count, char) for char, count in self.freq.items()]
        heapq.heapify(heap)

        result = []
        prev_count, prev_char = 0, ''

        while heap:
            count, char = heapq.heappop(heap)
            result.append(char)

            if prev_count < 0:
                heapq.heappush(heap, (prev_count, prev_char))

            prev_count = count + 1
            prev_char = char

        return ''.join(result)

    def count_beautiful_substrings(self):
        """Count beautiful substrings"""
        count = 0

        for i in range(self.n):
            for j in range(i + 1, self.n + 1):
                substring = self.s[i:j]

                is_beautiful = True
                for k in range(len(substring) - 1):
                    if substring[k] == substring[k + 1]:
                        is_beautiful = False
                        break

                if is_beautiful:
                    count += 1

        return count

    def get_adjacent_pairs(self):
        """Get all adjacent same character pairs"""
        pairs = []
        for i in range(self.n - 1):
            if self.s[i] == self.s[i + 1]:
                pairs.append((i, self.s[i]))
        return pairs

    def full_analysis(self):
        """Complete analysis"""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + "✨ BEAUTIFUL STRING ANALYSIS".center(58) + "║")
        print("╠" + "═" * 58 + "╣")

        s_display = self.s if len(self.s) <= 40 else self.s[:37] + "..."
        print(f"║  📝 Input: '{s_display}'".ljust(59) + "║")
        print(f"║  📏 Length: {self.n}".ljust(59) + "║")
        print("╠" + "═" * 58 + "╣")

        # Checks
        no_adj = self.is_beautiful_no_adjacent()
        unique = self.is_beautiful_unique()
        can_rearr = self.can_rearrange_beautiful()

        print(f"║  ✅ No Adjacent Same: {'✅ Yes' if no_adj else '❌ No'}".ljust(59) + "║")
        print(f"║  ✅ All Unique: {'✅ Yes' if unique else '❌ No'}".ljust(59) + "║")
        print(f"║  ✅ Can Rearrange: {'✅ Yes' if can_rearr else '❌ No'}".ljust(59) + "║")

        print("╠" + "═" * 58 + "╣")

        # Character frequency
        print("║  📊 Character Frequencies:".ljust(59) + "║")
        for char, count in sorted(self.freq.items())[:5]:
            bar = "█" * min(count, 20)
            print(f"║     '{char}': {count} {bar}".ljust(59) + "║")
        if len(self.freq) > 5:
            print(f"║     ... and {len(self.freq) - 5} more".ljust(59) + "║")

        print("╠" + "═" * 58 + "╣")

        # Operations
        min_del = self.min_deletions_no_adjacent()
        removed = self.make_beautiful_remove()
        rearranged = self.rearrange_beautiful()

        print(f"║  🔧 Min Deletions: {min_del}".ljust(59) + "║")
        print(f"║  🔧 By Removing: '{removed}'".ljust(59) + "║")

        if rearranged:
            rearr_display = rearranged if len(rearranged) <= 30 else rearranged[:27] + "..."
            print(f"║  🔧 By Rearranging: '{rearr_display}'".ljust(59) + "║")
        else:
            print(f"║  🔧 By Rearranging: Impossible".ljust(59) + "║")

        # Adjacent pairs
        pairs = self.get_adjacent_pairs()
        if pairs:
            print("╠" + "═" * 58 + "╣")
            print(f"║  ⚠️ Adjacent Same Pairs: {len(pairs)}".ljust(59) + "║")
            for pos, char in pairs[:5]:
                print(f"║     Position {pos}: '{char}{char}'".ljust(59) + "║")

        print("╚" + "═" * 58 + "╝")


# Usage
test_strings = [
    "aabbcc",
    "abab",
    "aaabbbccc",
    "beautiful",
]

for s in test_strings:
    analyzer = BeautifulString(s)
    analyzer.full_analysis()


# In[ ]:


s = "aabbcc"

# Check if beautiful (no adjacent same)
is_beautiful = all(s[i] != s[i+1] for i in range(len(s)-1))
print(f"Is Beautiful: {is_beautiful}")

# Remove adjacent duplicates
beautiful = ''.join(c for i, c in enumerate(s) if i == 0 or c != s[i-1])
print(f"Remove adjacent: {beautiful}")

# Min deletions
min_del = sum(1 for i in range(1, len(s)) if s[i] == s[i-1])
print(f"Min deletions: {min_del}")

# Can rearrange?
from collections import Counter
can_rearrange = max(Counter(s).values()) <= (len(s) + 1) // 2
print(f"Can rearrange: {can_rearrange}")

# Count adjacent pairs
adj_pairs = sum(1 for i in range(len(s)-1) if s[i] == s[i+1])
print(f"Adjacent pairs: {adj_pairs}")

