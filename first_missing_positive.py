#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def first_missing_positive_brute(arr):
    """
    Check each positive integer starting from 1
    Time: O(n²), Space: O(1)
    """
    positive = 1

    while True:
        if positive not in arr:
            return positive
        positive += 1

# Examples
test_cases = [
    [1, 2, 0],
    [3, 4, -1, 1],
    [7, 8, 9, 11, 12],
    [1, 2, 3, 4, 5],
    [-1, -2, -3],
    [1],
    [2],
]

print("🔍 FIRST MISSING POSITIVE (Brute Force)")
print("=" * 50)

for arr in test_cases:
    result = first_missing_positive_brute(arr)
    print(f"Array: {str(arr):<25} → Missing: {result}")


# In[ ]:


def first_missing_positive_sort(arr):
    """
    Sort array and find first missing
    Time: O(n log n), Space: O(1) or O(n)
    """
    # Filter positive numbers and sort
    positives = sorted(set(x for x in arr if x > 0))

    expected = 1

    for num in positives:
        if num != expected:
            return expected
        expected += 1

    return expected

# Example with visualization
arr = [3, 4, -1, 1, 5, 2]

print(f"Original Array: {arr}")

positives = sorted(set(x for x in arr if x > 0))
print(f"Filtered & Sorted: {positives}")

result = first_missing_positive_sort(arr)
print(f"First Missing Positive: {result}")


# In[ ]:


def first_missing_positive_set(arr):
    """
    Use set for O(1) lookup
    Time: O(n), Space: O(n)
    """
    num_set = set(arr)

    # Start checking from 1
    positive = 1

    while positive in num_set:
        positive += 1

    return positive

# Example
arr = [3, 4, -1, 1]

print(f"Array: {arr}")
print(f"Set: {set(arr)}")
print(f"First Missing Positive: {first_missing_positive_set(arr)}")


# In[ ]:


def first_missing_positive_optimal(arr):
    """
    Optimal solution using cyclic sort
    Time: O(n), Space: O(1)

    Key insight: 
    - Answer is always in range [1, n+1]
    - Place each number at its correct index (num at index num-1)
    """
    n = len(arr)

    # Step 1: Place each positive number at correct position
    # Number x should be at index x-1
    i = 0
    while i < n:
        # Current number
        num = arr[i]

        # Correct position for this number
        correct_pos = num - 1

        # Swap if:
        # 1. Number is positive
        # 2. Number is within range [1, n]
        # 3. Number is not already at correct position
        if 1 <= num <= n and arr[correct_pos] != num:
            arr[i], arr[correct_pos] = arr[correct_pos], arr[i]
        else:
            i += 1

    # Step 2: Find first position where arr[i] != i+1
    for i in range(n):
        if arr[i] != i + 1:
            return i + 1

    # All positions filled correctly
    return n + 1

# Example
arr = [3, 4, -1, 1]
arr_copy = arr.copy()

result = first_missing_positive_optimal(arr)

print(f"Original: {arr_copy}")
print(f"After rearrangement: {arr}")
print(f"First Missing Positive: {result}")


# In[ ]:


def first_missing_positive_visualized(arr):
    """
    Visualize the cyclic sort process
    """
    arr = arr.copy()
    n = len(arr)

    print("🔍 FINDING FIRST MISSING POSITIVE")
    print("=" * 55)
    print(f"Original Array: {arr}")
    print(f"Array Length (n): {n}")
    print(f"Valid range: [1, {n}]")
    print()
    print("📊 STEP 1: CYCLIC SORT (Place numbers at correct positions)")
    print("-" * 55)

    step = 0
    i = 0

    while i < n:
        num = arr[i]
        correct_pos = num - 1

        if 1 <= num <= n and arr[correct_pos] != num:
            step += 1
            print(f"Step {step}: arr[{i}]={num} should be at index {correct_pos}")
            print(f"         Swap arr[{i}] and arr[{correct_pos}]")
            arr[i], arr[correct_pos] = arr[correct_pos], arr[i]
            print(f"         Array: {arr}")
        else:
            if num < 1 or num > n:
                print(f"Skip: arr[{i}]={num} is out of range [1,{n}]")
            elif arr[correct_pos] == num:
                print(f"Skip: arr[{i}]={num} already at correct position")
            i += 1

    print()
    print("📊 STEP 2: FIND FIRST MISSING")
    print("-" * 55)
    print(f"Final Array: {arr}")
    print()
    print(f"{'Index':<8}{'Expected':<12}{'Actual':<10}{'Match'}")
    print("-" * 40)

    result = n + 1

    for i in range(n):
        expected = i + 1
        actual = arr[i]
        match = "✅" if expected == actual else "❌ MISSING!"
        print(f"{i:<8}{expected:<12}{actual:<10}{match}")

        if expected != actual and result == n + 1:
            result = expected

    print("-" * 55)
    print(f"✅ First Missing Positive: {result}")

    return result

# Example
arr = [3, 4, -1, 1]
first_missing_positive_visualized(arr)


# In[ ]:


def first_missing_positive_marking(arr):
    """
    Mark indices using negative numbers
    Time: O(n), Space: O(1)
    """
    n = len(arr)

    # Step 1: Replace negative numbers and zeros with n+1
    for i in range(n):
        if arr[i] <= 0:
            arr[i] = n + 1

    print(f"After replacing non-positives: {arr}")

    # Step 2: Mark indices
    for i in range(n):
        num = abs(arr[i])
        if num <= n:
            # Mark index num-1 as negative
            arr[num - 1] = -abs(arr[num - 1])

    print(f"After marking: {arr}")

    # Step 3: Find first positive (unmarked) index
    for i in range(n):
        if arr[i] > 0:
            return i + 1

    return n + 1

# Example
arr = [3, 4, -1, 1]
print(f"Original: {arr}")
result = first_missing_positive_marking(arr.copy())
print(f"First Missing Positive: {result}")


# In[ ]:


def first_missing_positive_safe(arr):
    """
    Handle all edge cases properly
    """
    # Edge case: Empty array
    if not arr:
        return 1, "Empty array"

    # Edge case: All negative
    if all(x <= 0 for x in arr):
        return 1, "All non-positive"

    # Edge case: All same positive
    if len(set(arr)) == 1 and arr[0] > 0:
        if arr[0] == 1:
            return 2, "All elements are 1"
        else:
            return 1, f"All elements are {arr[0]}"

    n = len(arr)
    arr = arr.copy()

    # Cyclic sort
    i = 0
    while i < n:
        num = arr[i]
        correct_pos = num - 1

        if 1 <= num <= n and arr[correct_pos] != num:
            arr[i], arr[correct_pos] = arr[correct_pos], arr[i]
        else:
            i += 1

    # Find missing
    for i in range(n):
        if arr[i] != i + 1:
            return i + 1, "Found in array"

    return n + 1, "All 1 to n present"

# Test all edge cases
test_cases = [
    [],                          # Empty
    [1],                         # Single 1
    [2],                         # Single non-1
    [-1, -2, -3],               # All negative
    [0, 0, 0],                  # All zeros
    [1, 1, 1, 1],               # All ones
    [5, 5, 5],                  # All same
    [1, 2, 3, 4, 5],            # Complete sequence
    [2, 3, 4, 5, 6],            # Missing 1
    [1, 2, 3, 5, 6],            # Missing 4
    [3, 4, -1, 1],              # Mixed
    [1, 2, 0],                  # With zero
    [7, 8, 9, 11, 12],          # Large numbers
    [1000000],                  # Very large
    list(range(1, 101)),        # 1 to 100
]

print("🔍 FIRST MISSING POSITIVE - ALL EDGE CASES")
print("=" * 65)

for arr in test_cases:
    result, reason = first_missing_positive_safe(arr)
    arr_str = str(arr)[:35] + "..." if len(str(arr)) > 35 else str(arr)
    print(f"{arr_str:<40} → {result} ({reason})")


# In[ ]:


def first_missing_positive_interactive():
    """Interactive first missing positive finder"""

    print("🔢 FIRST MISSING POSITIVE FINDER")
    print("=" * 45)

    # Get input
    user_input = input("Enter array elements (space separated): ")

    if not user_input.strip():
        print("❌ Empty input!")
        return

    arr = list(map(int, user_input.split()))

    print("\n📋 MENU")
    print("1. Find result (simple)")
    print("2. Find result (optimal)")
    print("3. Step-by-step visualization")
    print("4. Full analysis")

    choice = input("\nChoice (1-4): ")

    print("\n" + "=" * 45)

    if choice == '1':
        # Simple method
        num_set = set(arr)
        result = 1
        while result in num_set:
            result += 1

        print(f"Array: {arr}")
        print(f"✅ First Missing Positive: {result}")

    elif choice == '2':
        # Optimal method
        arr_copy = arr.copy()
        n = len(arr_copy)

        i = 0
        while i < n:
            num = arr_copy[i]
            correct = num - 1
            if 1 <= num <= n and arr_copy[correct] != num:
                arr_copy[i], arr_copy[correct] = arr_copy[correct], arr_copy[i]
            else:
                i += 1

        result = n + 1
        for i in range(n):
            if arr_copy[i] != i + 1:
                result = i + 1
                break

        print(f"Original: {arr}")
        print(f"Rearranged: {arr_copy}")
        print(f"✅ First Missing Positive: {result}")

    elif choice == '3':
        first_missing_positive_visualized(arr)

    elif choice == '4':
        print(f"📊 FULL ANALYSIS")
        print(f"Array: {arr}")
        print(f"Length: {len(arr)}")
        print()

        # Statistics
        positives = [x for x in arr if x > 0]
        negatives = [x for x in arr if x < 0]
        zeros = arr.count(0)

        print(f"Positive count: {len(positives)}")
        print(f"Negative count: {len(negatives)}")
        print(f"Zero count: {zeros}")
        print()

        if positives:
            print(f"Positive numbers: {sorted(set(positives))}")
            print(f"Min positive: {min(positives)}")
            print(f"Max positive: {max(positives)}")
        print()

        # Find result
        num_set = set(arr)
        result = 1
        while result in num_set:
            result += 1

        print(f"✅ First Missing Positive: {result}")

        # Show first few missing
        missing = []
        check = 1
        while len(missing) < 5:
            if check not in num_set:
                missing.append(check)
            check += 1
        print(f"📍 First 5 missing positives: {missing}")

# Run
first_missing_positive_interactive()


# In[ ]:


class FirstMissingPositive:
    """Complete utility for first missing positive"""

    def __init__(self, arr):
        self.original = arr.copy()
        self.arr = arr.copy()
        self.n = len(arr)

    def reset(self):
        """Reset to original array"""
        self.arr = self.original.copy()
        return self

    def find_brute(self):
        """Brute force O(n²)"""
        positive = 1
        while positive in self.arr:
            positive += 1
        return positive

    def find_set(self):
        """Using set O(n) time, O(n) space"""
        num_set = set(self.arr)
        positive = 1
        while positive in num_set:
            positive += 1
        return positive

    def find_optimal(self):
        """Cyclic sort O(n) time, O(1) space"""
        arr = self.arr.copy()
        n = len(arr)

        i = 0
        while i < n:
            num = arr[i]
            correct = num - 1
            if 1 <= num <= n and arr[correct] != num:
                arr[i], arr[correct] = arr[correct], arr[i]
            else:
                i += 1

        for i in range(n):
            if arr[i] != i + 1:
                return i + 1

        return n + 1

    def find_all_missing(self, count=10):
        """Find first 'count' missing positive integers"""
        num_set = set(self.arr)
        missing = []
        check = 1

        while len(missing) < count:
            if check not in num_set:
                missing.append(check)
            check += 1

        return missing

    def get_statistics(self):
        """Get array statistics"""
        positives = [x for x in self.arr if x > 0]

        return {
            'length': self.n,
            'positive_count': len(positives),
            'negative_count': sum(1 for x in self.arr if x < 0),
            'zero_count': self.arr.count(0),
            'unique_positives': sorted(set(positives)),
            'min_positive': min(positives) if positives else None,
            'max_positive': max(positives) if positives else None,
        }

    def is_complete_sequence(self, start=1):
        """Check if array contains complete sequence from start"""
        num_set = set(self.arr)
        i = start

        while i in num_set:
            i += 1

        return i - start, i  # (length of sequence, first missing)

    def visualize_solution(self):
        """Visualize the cyclic sort solution"""
        arr = self.arr.copy()
        n = len(arr)

        print("\n⚖️ CYCLIC SORT VISUALIZATION")
        print("=" * 50)
        print(f"Original: {arr}")
        print(f"Goal: Place number k at index k-1")
        print("-" * 50)

        steps = []
        i = 0

        while i < n:
            num = arr[i]
            correct = num - 1

            if 1 <= num <= n and arr[correct] != num:
                steps.append({
                    'action': 'swap',
                    'i': i,
                    'j': correct,
                    'before': arr.copy(),
                })
                arr[i], arr[correct] = arr[correct], arr[i]
                steps[-1]['after'] = arr.copy()
            else:
                i += 1

        for idx, step in enumerate(steps, 1):
            print(f"Step {idx}: Swap positions {step['i']} and {step['j']}")
            print(f"         {step['before']} → {step['after']}")

        print("-" * 50)
        print(f"Final: {arr}")

        # Find result
        result = n + 1
        for i in range(n):
            if arr[i] != i + 1:
                result = i + 1
                break

        print(f"\n✅ First Missing Positive: {result}")

        return result

    def full_analysis(self):
        """Complete analysis report"""
        stats = self.get_statistics()
        result = self.find_optimal()
        missing = self.find_all_missing(5)
        seq_len, first_miss = self.is_complete_sequence()

        print("\n" + "╔" + "═" * 55 + "╗")
        print("║" + "🔢 FIRST MISSING POSITIVE ANALYSIS".center(55) + "║")
        print("╠" + "═" * 55 + "╣")

        arr_str = str(self.original)
        if len(arr_str) > 45:
            arr_str = arr_str[:42] + "..."

        print(f"║  📝 Array: {arr_str}".ljust(56) + "║")
        print(f"║  📏 Length: {stats['length']}".ljust(56) + "║")
        print("╠" + "═" * 55 + "╣")

        print(f"║  ➕ Positive Numbers: {stats['positive_count']}".ljust(56) + "║")
        print(f"║  ➖ Negative Numbers: {stats['negative_count']}".ljust(56) + "║")
        print(f"║  0️⃣  Zeros: {stats['zero_count']}".ljust(56) + "║")

        if stats['unique_positives']:
            uniq_str = str(stats['unique_positives'])
            if len(uniq_str) > 35:
                uniq_str = uniq_str[:32] + "..."
            print(f"║  📊 Unique Positives: {uniq_str}".ljust(56) + "║")
            print(f"║  ⬇️  Min Positive: {stats['min_positive']}".ljust(56) + "║")
            print(f"║  ⬆️  Max Positive: {stats['max_positive']}".ljust(56) + "║")

        print("╠" + "═" * 55 + "╣")

        print(f"║  📈 Complete Sequence Length: {seq_len} (starting from 1)".ljust(56) + "║")
        print(f"║  📍 First 5 Missing: {missing}".ljust(56) + "║")

        print("╠" + "═" * 55 + "╣")
        print(f"║  ✅ FIRST MISSING POSITIVE: {result}".ljust(56) + "║")
        print("╚" + "═" * 55 + "╝")

        return result


# Usage
arr = [3, 4, -1, 1, 7, 2, 8]
finder = FirstMissingPositive(arr)

# Full analysis
finder.full_analysis()

# Visualize
print()
finder.visualize_solution()


# In[ ]:


import time
import random

def measure_time(func, arr, iterations=100):
    """Measure execution time"""
    start = time.time()
    for _ in range(iterations):
        func(arr.copy())
    end = time.time()
    return (end - start) / iterations * 1000

# Methods
def brute_force(arr):
    pos = 1
    while pos in arr:
        pos += 1
    return pos

def using_set(arr):
    s = set(arr)
    pos = 1
    while pos in s:
        pos += 1
    return pos

def sorting_method(arr):
    positives = sorted(set(x for x in arr if x > 0))
    expected = 1
    for num in positives:
        if num != expected:
            return expected
        expected += 1
    return expected

def cyclic_sort(arr):
    n = len(arr)
    i = 0
    while i < n:
        num = arr[i]
        correct = num - 1
        if 1 <= num <= n and arr[correct] != num:
            arr[i], arr[correct] = arr[correct], arr[i]
        else:
            i += 1
    for i in range(n):
        if arr[i] != i + 1:
            return i + 1
    return n + 1

# Generate test array
arr = list(range(1, 1001))  # 1 to 1000
arr.remove(500)  # Remove 500
random.shuffle(arr)
arr.extend([-1, -2, 0, 1001, 1002])  # Add some noise

print("⚡ PERFORMANCE COMPARISON")
print("=" * 50)
print(f"Array size: {len(arr)}")
print(f"Missing number: 500")
print("-" * 50)

methods = [
    ("Brute Force O(n²)", brute_force),
    ("Using Set O(n)", using_set),
    ("Sorting O(n log n)", sorting_method),
    ("Cyclic Sort O(n)", cyclic_sort),
]

for name, func in methods:
    result = func(arr.copy())
    time_ms = measure_time(func, arr)
    print(f"{name:<25}: {time_ms:.4f} ms (result: {result})")

print("-" * 50)
print("✅ Cyclic Sort is best for O(1) space!")
print("✅ Set method is best for simplicity!")


# In[ ]:


def find_all_missing_in_range(arr, start=1, end=None):
    """Find all missing positive integers in range"""
    if end is None:
        end = len(arr)

    present = set(arr)
    missing = [i for i in range(start, end + 1) if i not in present]

    return missing

# Example
arr = [4, 3, 2, 7, 8, 2, 3, 1]
missing = find_all_missing_in_range(arr)

print(f"Array: {arr}")
print(f"Range: 1 to {len(arr)}")
print(f"Missing: {missing}")


# In[ ]:


def find_duplicate_and_missing(arr):
    """
    Array of n integers from 1 to n
    One number is duplicate, one is missing
    """
    n = len(arr)

    # Expected sum and sum of squares
    expected_sum = n * (n + 1) // 2
    expected_sq_sum = n * (n + 1) * (2 * n + 1) // 6

    # Actual sums
    actual_sum = sum(arr)
    actual_sq_sum = sum(x * x for x in arr)

    # diff1 = missing - duplicate
    diff1 = expected_sum - actual_sum

    # diff2 = missing² - duplicate² = (missing + duplicate)(missing - duplicate)
    diff2 = expected_sq_sum - actual_sq_sum

    # missing + duplicate
    sum_md = diff2 // diff1

    missing = (diff1 + sum_md) // 2
    duplicate = sum_md - missing

    return duplicate, missing

# Example
arr = [1, 2, 2, 4, 5]  # 3 is missing, 2 is duplicate

dup, miss = find_duplicate_and_missing(arr)
print(f"Array: {arr}")
print(f"Duplicate: {dup}")
print(f"Missing: {miss}")


# In[ ]:


arr = [3, 4, -1, 1]

# Using set
first_missing = next(i for i in range(1, len(arr) + 2) if i not in set(arr))
print(f"Method 1: {first_missing}")

# Using while loop (compact)
s = set(arr); m = 1
while m in s: m += 1
print(f"Method 2: {m}")

# Using reduce
from functools import reduce
result = reduce(lambda m, _: m + 1 if m in set(arr) else m, range(len(arr) + 1), 1)
print(f"Method 3: {result}")

# Using next with generator
result = next((i for i in range(1, len(arr) + 2) if i not in arr), 1)
print(f"Method 4: {result}")

# Lambda function
find_missing = lambda a: next(i for i in range(1, len(a) + 2) if i not in set(a))
print(f"Lambda: {find_missing(arr)}")

