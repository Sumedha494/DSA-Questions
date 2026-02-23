#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def positives_first(arr):
    """
    Rearrange: all positives first, then all negatives
    Time: O(n), Space: O(n)
    """
    positives = [x for x in arr if x > 0]
    negatives = [x for x in arr if x < 0]
    zeros = [x for x in arr if x == 0]

    return positives + zeros + negatives

# Example
arr = [3, -2, 5, -1, 4, 0, -6, 2]

result = positives_first(arr)

print(f"Original: {arr}")
print(f"Rearranged: {result}")


# In[ ]:


def negatives_first(arr):
    """
    Rearrange: all negatives first, then all positives
    Time: O(n), Space: O(n)
    """
    negatives = [x for x in arr if x < 0]
    zeros = [x for x in arr if x == 0]
    positives = [x for x in arr if x > 0]

    return negatives + zeros + positives

# Example
arr = [3, -2, 5, -1, 4, 0, -6, 2]

result = negatives_first(arr)

print(f"Original: {arr}")
print(f"Rearranged: {result}")


# In[ ]:


def rearrange_inplace(arr):
    """
    Rearrange in-place: positives first, then negatives
    Time: O(n), Space: O(1)
    Similar to partition in QuickSort
    """
    arr = arr.copy()
    n = len(arr)

    # Index to place next positive number
    pos_idx = 0

    for i in range(n):
        # If positive number found, swap with pos_idx
        if arr[i] > 0:
            arr[i], arr[pos_idx] = arr[pos_idx], arr[i]
            pos_idx += 1

    return arr

# Example
arr = [3, -2, 5, -1, 4, -6, 2]

print(f"Original: {arr}")
print(f"In-place: {rearrange_inplace(arr)}")


# In[ ]:


def rearrange_alternating(arr):
    """
    Rearrange: positive at even index, negative at odd index
    [pos, neg, pos, neg, ...]

    Assumption: Equal number of positive and negative numbers
    Time: O(n), Space: O(n)
    """
    positives = [x for x in arr if x > 0]
    negatives = [x for x in arr if x < 0]

    result = []
    pos_idx = 0
    neg_idx = 0

    for i in range(len(arr)):
        if i % 2 == 0:  # Even index - positive
            if pos_idx < len(positives):
                result.append(positives[pos_idx])
                pos_idx += 1
            elif neg_idx < len(negatives):
                result.append(negatives[neg_idx])
                neg_idx += 1
        else:  # Odd index - negative
            if neg_idx < len(negatives):
                result.append(negatives[neg_idx])
                neg_idx += 1
            elif pos_idx < len(positives):
                result.append(positives[pos_idx])
                pos_idx += 1

    return result

# Example
arr = [3, -2, 5, -1, 4, -6]

result = rearrange_alternating(arr)

print(f"Original: {arr}")
print(f"Alternating: {result}")
print(f"Pattern: [+, -, +, -, +, -]")


# In[ ]:


def rearrange_alternating_ordered(arr):
    """
    Alternating positive and negative while maintaining relative order
    Time: O(n), Space: O(n)
    """
    positives = [x for x in arr if x > 0]
    negatives = [x for x in arr if x < 0]

    result = []
    pos_idx = 0
    neg_idx = 0
    n = len(arr)

    # Place alternately
    for i in range(n):
        if i % 2 == 0:  # Even - positive
            if pos_idx < len(positives):
                result.append(positives[pos_idx])
                pos_idx += 1
        else:  # Odd - negative
            if neg_idx < len(negatives):
                result.append(negatives[neg_idx])
                neg_idx += 1

    # Append remaining elements
    while pos_idx < len(positives):
        result.append(positives[pos_idx])
        pos_idx += 1

    while neg_idx < len(negatives):
        result.append(negatives[neg_idx])
        neg_idx += 1

    return result

# Example
arr = [1, 2, 3, -4, -1, 4]

result = rearrange_alternating_ordered(arr)

print(f"Original: {arr}")
print(f"Positives: {[x for x in arr if x > 0]}")
print(f"Negatives: {[x for x in arr if x < 0]}")
print(f"Alternating (ordered): {result}")


# In[ ]:


def rearrange_visualized(arr):
    """
    Visualize the rearrangement process
    """
    print("🔄 REARRANGE BY SIGN - VISUALIZATION")
    print("=" * 55)
    print(f"Original Array: {arr}")
    print()

    # Separate positives and negatives
    positives = []
    negatives = []

    for x in arr:
        if x > 0:
            positives.append(x)
            print(f"  {x:>3} → ➕ Positive list: {positives}")
        elif x < 0:
            negatives.append(x)
            print(f"  {x:>3} → ➖ Negative list: {negatives}")
        else:
            print(f"  {x:>3} → 0️⃣  Zero (ignored)")

    print()
    print(f"➕ Positives: {positives}")
    print(f"➖ Negatives: {negatives}")
    print()

    # Rearrange alternating
    result = []
    pos_idx = neg_idx = 0

    print("📊 Building alternating array:")
    print("-" * 40)

    for i in range(len(arr)):
        if i % 2 == 0:  # Even - positive
            if pos_idx < len(positives):
                result.append(positives[pos_idx])
                print(f"  Index {i} (even) → +{positives[pos_idx]} → {result}")
                pos_idx += 1
        else:  # Odd - negative
            if neg_idx < len(negatives):
                result.append(negatives[neg_idx])
                print(f"  Index {i} (odd)  → {negatives[neg_idx]} → {result}")
                neg_idx += 1

    # Append remaining
    while pos_idx < len(positives):
        result.append(positives[pos_idx])
        print(f"  Remaining → +{positives[pos_idx]} → {result}")
        pos_idx += 1

    while neg_idx < len(negatives):
        result.append(negatives[neg_idx])
        print(f"  Remaining → {negatives[neg_idx]} → {result}")
        neg_idx += 1

    print("-" * 40)
    print(f"✅ Final Result: {result}")

    return result

# Example
arr = [3, -2, 5, -1, 4, -6, 7]
rearrange_visualized(arr)


# In[ ]:


def rearrange_alternating_inplace(arr):
    """
    In-place alternating rearrangement

    Step 1: Move all positives to left
    Step 2: Swap alternating positions

    Time: O(n), Space: O(1)
    Note: Does not preserve relative order
    """
    arr = arr.copy()
    n = len(arr)

    # Step 1: Partition - positives left, negatives right
    pos_idx = 0
    for i in range(n):
        if arr[i] > 0:
            arr[i], arr[pos_idx] = arr[pos_idx], arr[i]
            pos_idx += 1

    print(f"After partition: {arr}")
    print(f"Positives: {arr[:pos_idx]}, Negatives: {arr[pos_idx:]}")

    # Step 2: Swap negatives to odd positions
    pos = 0  # First positive (even position)
    neg = pos_idx  # First negative

    while pos < n and neg < n and arr[pos] > 0:
        arr[pos + 1], arr[neg] = arr[neg], arr[pos + 1]
        pos += 2
        neg += 1

    return arr

# Example
arr = [3, -2, 5, -1, 4, -6]

print(f"Original: {arr}")
result = rearrange_alternating_inplace(arr)
print(f"Result: {result}")


# In[ ]:


def rearrange_unequal(arr, start_with='positive'):
    """
    Handle arrays with unequal positive and negative counts
    Extra elements go at the end
    """
    positives = [x for x in arr if x > 0]
    negatives = [x for x in arr if x < 0]
    zeros = [x for x in arr if x == 0]

    result = []
    pos_idx = neg_idx = 0

    # Determine order
    if start_with == 'positive':
        first, second = positives, negatives
        first_idx, second_idx = 0, 0
    else:
        first, second = negatives, positives
        first_idx, second_idx = 0, 0

    # Alternate
    while first_idx < len(first) and second_idx < len(second):
        result.append(first[first_idx])
        first_idx += 1
        result.append(second[second_idx])
        second_idx += 1

    # Append remaining
    while first_idx < len(first):
        result.append(first[first_idx])
        first_idx += 1

    while second_idx < len(second):
        result.append(second[second_idx])
        second_idx += 1

    # Append zeros
    result.extend(zeros)

    return result

# Examples
test_cases = [
    [1, -2, 3, -4, 5, -6],        # Equal counts
    [1, 2, 3, 4, -1, -2],         # More positives
    [-1, -2, -3, -4, 1, 2],       # More negatives
    [1, 2, 3, 0, -1, 0, -2],      # With zeros
]

print("🔄 REARRANGE WITH UNEQUAL COUNTS")
print("=" * 55)

for arr in test_cases:
    pos_count = sum(1 for x in arr if x > 0)
    neg_count = sum(1 for x in arr if x < 0)
    zero_count = arr.count(0)

    result = rearrange_unequal(arr)

    print(f"\nOriginal: {arr}")
    print(f"Counts: +{pos_count}, -{neg_count}, 0:{zero_count}")
    print(f"Result: {result}")


# In[ ]:


class SignRearranger:
    """Multiple rearrangement patterns"""

    def __init__(self, arr):
        self.arr = arr.copy()
        self.positives = [x for x in arr if x > 0]
        self.negatives = [x for x in arr if x < 0]
        self.zeros = [x for x in arr if x == 0]

    def positives_first(self):
        """[+, +, +, ..., -, -, -]"""
        return self.positives + self.zeros + self.negatives

    def negatives_first(self):
        """[-, -, -, ..., +, +, +]"""
        return self.negatives + self.zeros + self.positives

    def alternating_pos_first(self):
        """[+, -, +, -, ...]"""
        result = []
        p, n = 0, 0

        for i in range(len(self.arr)):
            if i % 2 == 0 and p < len(self.positives):
                result.append(self.positives[p])
                p += 1
            elif n < len(self.negatives):
                result.append(self.negatives[n])
                n += 1
            elif p < len(self.positives):
                result.append(self.positives[p])
                p += 1

        result.extend(self.zeros)
        return result

    def alternating_neg_first(self):
        """[-, +, -, +, ...]"""
        result = []
        p, n = 0, 0

        for i in range(len(self.arr)):
            if i % 2 == 0 and n < len(self.negatives):
                result.append(self.negatives[n])
                n += 1
            elif p < len(self.positives):
                result.append(self.positives[p])
                p += 1
            elif n < len(self.negatives):
                result.append(self.negatives[n])
                n += 1

        result.extend(self.zeros)
        return result

    def zigzag(self):
        """[max, min, max, min, ...]"""
        sorted_arr = sorted(self.arr)
        result = []
        left, right = 0, len(sorted_arr) - 1

        while left <= right:
            if left != right:
                result.append(sorted_arr[right])
                result.append(sorted_arr[left])
            else:
                result.append(sorted_arr[left])
            right -= 1
            left += 1

        return result

    def show_all_patterns(self):
        """Display all rearrangement patterns"""
        print("\n" + "╔" + "═" * 55 + "╗")
        print("║" + "🔄 ALL REARRANGEMENT PATTERNS".center(55) + "║")
        print("╠" + "═" * 55 + "╣")
        print(f"║  Original: {self.arr}".ljust(56) + "║")
        print("╠" + "═" * 55 + "╣")
        print(f"║  ➕ Positives First:".ljust(56) + "║")
        print(f"║     {self.positives_first()}".ljust(56) + "║")
        print(f"║  ➖ Negatives First:".ljust(56) + "║")
        print(f"║     {self.negatives_first()}".ljust(56) + "║")
        print(f"║  🔀 Alternating (+, -, +, -):".ljust(56) + "║")
        print(f"║     {self.alternating_pos_first()}".ljust(56) + "║")
        print(f"║  🔀 Alternating (-, +, -, +):".ljust(56) + "║")
        print(f"║     {self.alternating_neg_first()}".ljust(56) + "║")
        print(f"║  📈 Zigzag (max, min, max, min):".ljust(56) + "║")
        print(f"║     {self.zigzag()}".ljust(56) + "║")
        print("╚" + "═" * 55 + "╝")


# Usage
arr = [3, -2, 5, -1, 4, -6, 0, 7]
rearranger = SignRearranger(arr)
rearranger.show_all_patterns()


# In[ ]:


def rearrangeArray(nums):
    """
    LeetCode 2149: Rearrange Array Elements by Sign

    - Equal number of positive and negative integers
    - Every consecutive pair has opposite signs
    - Starts with positive
    - Order of same-sign integers is preserved

    Time: O(n), Space: O(n)
    """
    n = len(nums)
    result = [0] * n

    pos_idx = 0  # Even indices for positives
    neg_idx = 1  # Odd indices for negatives

    for num in nums:
        if num > 0:
            result[pos_idx] = num
            pos_idx += 2
        else:
            result[neg_idx] = num
            neg_idx += 2

    return result

# Example (LeetCode)
nums = [3, 1, -2, -5, 2, -4]

result = rearrangeArray(nums)

print(f"Input: {nums}")
print(f"Output: {result}")
print()

# Verify
print("Verification:")
for i, num in enumerate(result):
    sign = "+" if num > 0 else "-"
    expected = "+" if i % 2 == 0 else "-"
    status = "✅" if sign == expected else "❌"
    print(f"  Index {i}: {num:>3} ({sign}) - Expected: {expected} {status}")


# In[ ]:


def rearrange_interactive():
    """Interactive rearrangement tool"""

    print("🔄 REARRANGE ARRAY BY SIGN")
    print("=" * 45)

    # Get input
    user_input = input("Enter array elements (space separated): ")
    arr = list(map(int, user_input.split()))

    # Count
    positives = [x for x in arr if x > 0]
    negatives = [x for x in arr if x < 0]
    zeros = [x for x in arr if x == 0]

    print(f"\n📊 Analysis:")
    print(f"   Positives ({len(positives)}): {positives}")
    print(f"   Negatives ({len(negatives)}): {negatives}")
    print(f"   Zeros ({len(zeros)}): {zeros}")

    print("\n📋 Select Pattern:")
    print("1. Positives First [+, +, ..., -, -]")
    print("2. Negatives First [-, -, ..., +, +]")
    print("3. Alternating [+, -, +, -, ...]")
    print("4. Alternating [-, +, -, +, ...]")
    print("5. Show All Patterns")

    choice = input("\nChoice (1-5): ")

    print("\n" + "=" * 45)

    if choice == '1':
        result = positives + zeros + negatives
        print(f"Pattern: Positives First")
        print(f"Result: {result}")

    elif choice == '2':
        result = negatives + zeros + positives
        print(f"Pattern: Negatives First")
        print(f"Result: {result}")

    elif choice == '3':
        result = []
        p, n = 0, 0
        for i in range(len(arr)):
            if i % 2 == 0 and p < len(positives):
                result.append(positives[p])
                p += 1
            elif n < len(negatives):
                result.append(negatives[n])
                n += 1
        while p < len(positives):
            result.append(positives[p])
            p += 1
        while n < len(negatives):
            result.append(negatives[n])
            n += 1
        result.extend(zeros)

        print(f"Pattern: Alternating [+, -, +, -, ...]")
        print(f"Result: {result}")

    elif choice == '4':
        result = []
        p, n = 0, 0
        for i in range(len(arr)):
            if i % 2 == 0 and n < len(negatives):
                result.append(negatives[n])
                n += 1
            elif p < len(positives):
                result.append(positives[p])
                p += 1
        while n < len(negatives):
            result.append(negatives[n])
            n += 1
        while p < len(positives):
            result.append(positives[p])
            p += 1
        result.extend(zeros)

        print(f"Pattern: Alternating [-, +, -, +, ...]")
        print(f"Result: {result}")

    elif choice == '5':
        print(f"Original: {arr}")
        print(f"Positives First: {positives + zeros + negatives}")
        print(f"Negatives First: {negatives + zeros + positives}")

        # Alternating +, -, ...
        alt1 = []
        p, n = 0, 0
        for i in range(len(positives) + len(negatives)):
            if i % 2 == 0 and p < len(positives):
                alt1.append(positives[p])
                p += 1
            elif n < len(negatives):
                alt1.append(negatives[n])
                n += 1
        print(f"Alternating [+,-]: {alt1 + zeros}")

# Run
rearrange_interactive()


# In[ ]:


class ArrayRearrangerBySign:
    """Complete utility for sign-based rearrangement"""

    def __init__(self, arr):
        self.original = arr.copy()
        self.arr = arr.copy()
        self._separate()

    def _separate(self):
        """Separate elements by sign"""
        self.positives = [x for x in self.arr if x > 0]
        self.negatives = [x for x in self.arr if x < 0]
        self.zeros = [x for x in self.arr if x == 0]

    def reset(self):
        """Reset to original"""
        self.arr = self.original.copy()
        self._separate()
        return self

    def get_stats(self):
        """Get statistics"""
        return {
            'length': len(self.arr),
            'positive_count': len(self.positives),
            'negative_count': len(self.negatives),
            'zero_count': len(self.zeros),
            'positives': self.positives,
            'negatives': self.negatives,
        }

    def positives_first(self):
        """All positives, then zeros, then negatives"""
        return self.positives + self.zeros + self.negatives

    def negatives_first(self):
        """All negatives, then zeros, then positives"""
        return self.negatives + self.zeros + self.positives

    def alternating(self, start='positive', handle_extra='end'):
        """
        Alternating arrangement
        start: 'positive' or 'negative'
        handle_extra: 'end' (append at end) or 'intersperse'
        """
        if start == 'positive':
            first, second = self.positives.copy(), self.negatives.copy()
        else:
            first, second = self.negatives.copy(), self.positives.copy()

        result = []

        while first and second:
            result.append(first.pop(0))
            result.append(second.pop(0))

        # Handle remaining
        result.extend(first)
        result.extend(second)
        result.extend(self.zeros)

        return result

    def sorted_by_sign(self, ascending=True):
        """Sort positives and negatives separately"""
        pos_sorted = sorted(self.positives, reverse=not ascending)
        neg_sorted = sorted(self.negatives, reverse=not ascending)

        return pos_sorted + self.zeros + neg_sorted

    def wave_pattern(self):
        """Wave pattern: larger, smaller, larger, smaller, ..."""
        sorted_arr = sorted(self.arr)
        result = []

        for i in range(0, len(sorted_arr), 2):
            if i + 1 < len(sorted_arr):
                result.append(sorted_arr[i + 1])
                result.append(sorted_arr[i])
            else:
                result.append(sorted_arr[i])

        return result

    def dutch_flag(self, pivot=0):
        """
        Dutch National Flag: negatives < pivot < positives
        """
        less = [x for x in self.arr if x < pivot]
        equal = [x for x in self.arr if x == pivot]
        greater = [x for x in self.arr if x > pivot]

        return less + equal + greater

    def validate_alternating(self, arr):
        """Check if array is alternating"""
        if len(arr) <= 1:
            return True, "Too short to validate"

        first_sign = 1 if arr[0] > 0 else -1

        for i in range(len(arr)):
            expected_sign = first_sign if i % 2 == 0 else -first_sign
            actual_sign = 1 if arr[i] > 0 else (-1 if arr[i] < 0 else 0)

            if arr[i] != 0 and actual_sign != expected_sign:
                return False, f"Violation at index {i}"

        return True, "Valid alternating pattern"

    def full_analysis(self):
        """Complete analysis and all patterns"""
        stats = self.get_stats()

        print("\n" + "╔" + "═" * 60 + "╗")
        print("║" + "🔄 ARRAY REARRANGEMENT BY SIGN - ANALYSIS".center(60) + "║")
        print("╠" + "═" * 60 + "╣")

        arr_str = str(self.original)
        if len(arr_str) > 50:
            arr_str = arr_str[:47] + "..."

        print(f"║  📝 Original: {arr_str}".ljust(61) + "║")
        print(f"║  📏 Length: {stats['length']}".ljust(61) + "║")
        print("╠" + "═" * 60 + "╣")

        print(f"║  ➕ Positives ({stats['positive_count']}): {stats['positives']}".ljust(61) + "║")
        print(f"║  ➖ Negatives ({stats['negative_count']}): {stats['negatives']}".ljust(61) + "║")
        print(f"║  0️⃣  Zeros ({stats['zero_count']})".ljust(61) + "║")
        print("╠" + "═" * 60 + "╣")

        print("║  📊 REARRANGEMENT PATTERNS:".ljust(61) + "║")
        print(f"║     1. Positives First: {self.positives_first()}".ljust(61) + "║")
        print(f"║     2. Negatives First: {self.negatives_first()}".ljust(61) + "║")
        print(f"║     3. Alternating [+,-]: {self.alternating('positive')}".ljust(61) + "║")
        print(f"║     4. Alternating [-,+]: {self.alternating('negative')}".ljust(61) + "║")
        print(f"║     5. Wave Pattern: {self.wave_pattern()}".ljust(61) + "║")
        print(f"║     6. Dutch Flag (pivot=0): {self.dutch_flag(0)}".ljust(61) + "║")

        print("╚" + "═" * 60 + "╝")


# Usage
arr = [3, -2, 5, -1, 4, -6, 0, 7, -8]
rearranger = ArrayRearrangerBySign(arr)
rearranger.full_analysis()


# In[ ]:


import time
import random

def measure_time(func, arr, iterations=1000):
    """Measure execution time"""
    start = time.time()
    for _ in range(iterations):
        func(arr.copy())
    end = time.time()
    return (end - start) / iterations * 1000

# Methods
def method_list_comp(arr):
    pos = [x for x in arr if x > 0]
    neg = [x for x in arr if x < 0]
    return pos + neg

def method_filter(arr):
    pos = list(filter(lambda x: x > 0, arr))
    neg = list(filter(lambda x: x < 0, arr))
    return pos + neg

def method_inplace(arr):
    pos_idx = 0
    for i in range(len(arr)):
        if arr[i] > 0:
            arr[i], arr[pos_idx] = arr[pos_idx], arr[i]
            pos_idx += 1
    return arr

def method_single_pass(arr):
    result = [0] * len(arr)
    pos, neg = 0, len(arr) - 1
    for x in arr:
        if x > 0:
            result[pos] = x
            pos += 1
        else:
            result[neg] = x
            neg -= 1
    return result

# Generate test array
arr = [random.randint(-100, 100) for _ in range(1000)]

print("⚡ PERFORMANCE COMPARISON")
print("=" * 50)
print(f"Array size: {len(arr)}")
print("-" * 50)

methods = [
    ("List Comprehension", method_list_comp),
    ("Filter", method_filter),
    ("In-place Swap", method_inplace),
    ("Single Pass", method_single_pass),
]

for name, func in methods:
    time_ms = measure_time(func, arr)
    print(f"{name:<20}: {time_ms:.4f} ms")

print("-" * 50)
print("✅ List Comprehension is most Pythonic!")
print("✅ In-place is best for O(1) space!")


# In[ ]:


arr = [3, -2, 5, -1, 4, -6]

# Positives first
pos_first = [x for x in arr if x > 0] + [x for x in arr if x <= 0]
print(f"Positives first: {pos_first}")

# Negatives first
neg_first = [x for x in arr if x < 0] + [x for x in arr if x >= 0]
print(f"Negatives first: {neg_first}")

# Using sorted with key
by_sign = sorted(arr, key=lambda x: (x <= 0, x))
print(f"Sorted by sign: {by_sign}")

# Alternating (equal counts)
from itertools import chain
pos = [x for x in arr if x > 0]
neg = [x for x in arr if x < 0]
alternating = list(chain.from_iterable(zip(pos, neg)))
print(f"Alternating: {alternating}")

# Using reduce
from functools import reduce
separated = reduce(lambda acc, x: (acc[0] + [x], acc[1]) if x > 0 else (acc[0], acc[1] + [x]), arr, ([], []))
print(f"Separated: {separated[0] + separated[1]}")

