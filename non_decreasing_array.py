#!/usr/bin/env python
# coding: utf-8

# In[1]:


def is_non_decreasing(arr):
    """
    Check if array is non-decreasing
    Time: O(n), Space: O(1)
    """
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True

# Examples
test_arrays = [
    [1, 2, 3, 4, 5],      # Strictly increasing
    [1, 2, 2, 3, 4],      # Non-decreasing with equal
    [1, 1, 1, 1, 1],      # All same
    [5, 4, 3, 2, 1],      # Decreasing
    [1, 3, 2, 4, 5],      # Has one dip
    [1],                   # Single element
    [],                    # Empty
]

print("🔍 CHECK NON-DECREASING ARRAY")
print("=" * 45)

for arr in test_arrays:
    result = is_non_decreasing(arr)
    status = "✅ Yes" if result else "❌ No"
    print(f"{str(arr):<25} → {status}")


# In[2]:


# Method 1: Using all() and zip()
def is_non_decreasing_v1(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

# Method 2: Using zip (more Pythonic)
def is_non_decreasing_v2(arr):
    return all(a <= b for a, b in zip(arr, arr[1:]))

# Method 3: Compare with sorted
def is_non_decreasing_v3(arr):
    return arr == sorted(arr)

# Method 4: Using NumPy
import numpy as np
def is_non_decreasing_v4(arr):
    return np.all(np.diff(arr) >= 0)

# Example
arr = [1, 2, 2, 3, 4, 5]

print(f"Array: {arr}")
print(f"Method 1 (loop):   {is_non_decreasing_v1(arr)}")
print(f"Method 2 (zip):    {is_non_decreasing_v2(arr)}")
print(f"Method 3 (sorted): {is_non_decreasing_v3(arr)}")
print(f"Method 4 (numpy):  {is_non_decreasing_v4(arr)}")


# In[3]:


def find_violations(arr):
    """Find all positions where array decreases"""
    violations = []

    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            violations.append({
                'index': i,
                'pair': (arr[i], arr[i + 1]),
                'diff': arr[i] - arr[i + 1]
            })

    return violations

# Example
arr = [1, 5, 3, 7, 4, 8, 2]

violations = find_violations(arr)

print(f"Array: {arr}")
print(f"\n📍 Violations Found: {len(violations)}")
print("-" * 40)

for v in violations:
    print(f"Index {v['index']}: {v['pair'][0]} > {v['pair'][1]} (diff: {v['diff']})")


# In[4]:


def can_be_non_decreasing(arr):
    """
    Check if array can become non-decreasing 
    by modifying at most ONE element

    Time: O(n), Space: O(1)
    """
    if len(arr) <= 2:
        return True

    violations = 0

    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            violations += 1

            if violations > 1:
                return False

            # Check if we can fix by modifying arr[i] or arr[i+1]
            # Case 1: Modify arr[i] to be <= arr[i+1]
            # Case 2: Modify arr[i+1] to be >= arr[i]

            # If neither works, return False
            if i > 0 and arr[i - 1] > arr[i + 1]:
                # Can't lower arr[i], must raise arr[i+1]
                if i + 2 < len(arr) and arr[i] > arr[i + 2]:
                    return False

    return True

# Test cases
test_cases = [
    [4, 2, 3],           # True: change 4 to 1
    [4, 2, 1],           # False: need 2 changes
    [3, 4, 2, 3],        # False: need 2 changes
    [1, 2, 3, 4, 5],     # True: already non-decreasing
    [5, 7, 1, 8],        # True: change 1 to 7
    [1, 1, 1],           # True: already non-decreasing
    [1, 2, 3, 2, 4],     # True: change 2 to 3
]

print("🔍 CAN MAKE NON-DECREASING (1 CHANGE)?")
print("=" * 50)

for arr in test_cases:
    result = can_be_non_decreasing(arr)
    status = "✅ Yes" if result else "❌ No"
    print(f"{str(arr):<25} → {status}")


# In[5]:


def make_non_decreasing(arr):
    """
    Make array non-decreasing with minimum changes
    Returns modified array and count of changes
    """
    arr = arr.copy()
    changes = []

    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            old_value = arr[i]
            arr[i] = arr[i - 1]  # Make equal to previous
            changes.append({
                'index': i,
                'old': old_value,
                'new': arr[i]
            })

    return arr, changes

# Example
arr = [5, 3, 4, 2, 7, 1]

result, changes = make_non_decreasing(arr)

print(f"Original: {arr}")
print(f"Modified: {result}")
print(f"\n📝 Changes Made: {len(changes)}")
print("-" * 35)

for c in changes:
    print(f"Index {c['index']}: {c['old']} → {c['new']}")


# In[6]:


def make_non_decreasing_optimal(arr):
    """
    Make non-decreasing by either:
    1. Increasing current element
    2. Decreasing previous element
    Choose option with minimum change
    """
    arr = arr.copy()
    changes = []

    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            # Option 1: Increase arr[i] to arr[i-1]
            increase_cost = arr[i - 1] - arr[i]

            # Option 2: Decrease arr[i-1] to arr[i]
            # (only if it doesn't violate with arr[i-2])
            can_decrease = (i == 1) or (arr[i] >= arr[i - 2])

            if can_decrease:
                # Decrease previous
                changes.append({
                    'action': 'decrease',
                    'index': i - 1,
                    'old': arr[i - 1],
                    'new': arr[i]
                })
                arr[i - 1] = arr[i]
            else:
                # Increase current
                changes.append({
                    'action': 'increase',
                    'index': i,
                    'old': arr[i],
                    'new': arr[i - 1]
                })
                arr[i] = arr[i - 1]

    return arr, changes

# Example
arr = [4, 2, 3]

result, changes = make_non_decreasing_optimal(arr)

print(f"Original: {arr}")
print(f"Modified: {result}")
print(f"\n📝 Changes Made:")

for c in changes:
    print(f"  {c['action'].title()} index {c['index']}: {c['old']} → {c['new']}")


# In[7]:


def visualize_non_decreasing_check(arr):
    """Visualize the checking process"""

    print(f"🔍 CHECKING: {arr}")
    print("=" * 50)

    if len(arr) <= 1:
        print("✅ Array has 0 or 1 element - Always non-decreasing!")
        return True

    is_valid = True

    for i in range(len(arr) - 1):
        current = arr[i]
        next_val = arr[i + 1]

        comparison = "<=" if current <= next_val else ">"
        status = "✅" if current <= next_val else "❌"

        print(f"Index {i}: {current} {comparison} {next_val} {status}")

        if current > next_val:
            is_valid = False

    print("-" * 50)
    if is_valid:
        print("✅ Array IS non-decreasing!")
    else:
        print("❌ Array is NOT non-decreasing!")

    return is_valid

# Examples
visualize_non_decreasing_check([1, 2, 2, 3, 4])
print()
visualize_non_decreasing_check([1, 5, 3, 4, 2])


# In[8]:


def min_changes_to_non_decreasing(arr):
    """
    Count minimum elements to change to make non-decreasing
    Using Longest Non-Decreasing Subsequence (LIS variant)

    Answer = n - length of longest non-decreasing subsequence
    """
    if len(arr) <= 1:
        return 0

    n = len(arr)

    # dp[i] = length of longest non-decreasing subsequence ending at i
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[j] <= arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    longest = max(dp)
    min_changes = n - longest

    return min_changes, longest

# Examples
test_arrays = [
    [1, 2, 3, 4, 5],       # 0 changes needed
    [5, 4, 3, 2, 1],       # 4 changes needed
    [1, 3, 2, 4, 3, 5],    # 2 changes needed
    [3, 1, 2, 1, 3, 1],    # 3 changes needed
]

print("📊 MINIMUM CHANGES TO MAKE NON-DECREASING")
print("=" * 55)

for arr in test_arrays:
    changes, longest = min_changes_to_non_decreasing(arr)
    print(f"Array: {arr}")
    print(f"  Longest non-decreasing subsequence: {longest}")
    print(f"  Minimum changes needed: {changes}")
    print()


# In[9]:


def longest_non_decreasing_subsequence(arr):
    """
    Find the actual longest non-decreasing subsequence
    """
    if not arr:
        return []

    n = len(arr)
    dp = [1] * n
    parent = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if arr[j] <= arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    # Find index of maximum length
    max_length = max(dp)
    max_index = dp.index(max_length)

    # Reconstruct subsequence
    subsequence = []
    idx = max_index

    while idx != -1:
        subsequence.append(arr[idx])
        idx = parent[idx]

    subsequence.reverse()

    return subsequence, max_length

# Example
arr = [3, 1, 2, 1, 4, 2, 5, 3, 6]

subseq, length = longest_non_decreasing_subsequence(arr)

print(f"Array: {arr}")
print(f"\n📈 Longest Non-Decreasing Subsequence:")
print(f"   {subseq}")
print(f"   Length: {length}")
print(f"\n📊 Elements to change: {len(arr) - length}")


# In[10]:


def make_non_decreasing_constrained(arr, max_increase=None, max_decrease=None):
    """
    Make non-decreasing with constraints on how much 
    each element can be changed
    """
    arr = arr.copy()
    changes = []
    failed = []

    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            needed_increase = arr[i - 1] - arr[i]

            # Try to increase current
            if max_increase is None or needed_increase <= max_increase:
                changes.append({
                    'index': i,
                    'old': arr[i],
                    'new': arr[i - 1],
                    'change': needed_increase
                })
                arr[i] = arr[i - 1]

            # Try to decrease previous
            elif max_decrease is None or needed_increase <= max_decrease:
                # Check if decreasing previous is valid
                if i == 1 or arr[i] >= arr[i - 2]:
                    changes.append({
                        'index': i - 1,
                        'old': arr[i - 1],
                        'new': arr[i],
                        'change': needed_increase
                    })
                    arr[i - 1] = arr[i]
                else:
                    failed.append(i)
            else:
                failed.append(i)

    success = len(failed) == 0

    return arr, changes, failed, success

# Example
arr = [5, 2, 8, 3, 9]

print("🔧 CONSTRAINED NON-DECREASING")
print("=" * 45)
print(f"Original: {arr}")
print()

# Without constraints
result1, changes1, _, _ = make_non_decreasing_constrained(arr)
print(f"No constraints: {result1}")

# With max increase of 2
result2, changes2, failed2, success2 = make_non_decreasing_constrained(arr, max_increase=2)
print(f"Max increase 2: {result2} {'✅' if success2 else '❌'}")
if failed2:
    print(f"  Failed at indices: {failed2}")


# In[11]:


def non_decreasing_analyzer():
    """Interactive non-decreasing array analyzer"""

    print("📈 NON-DECREASING ARRAY ANALYZER")
    print("=" * 45)

    # Get input
    user_input = input("Enter array elements (space separated): ")
    arr = list(map(int, user_input.split()))

    print("\n📋 MENU")
    print("1. Check if non-decreasing")
    print("2. Find violations")
    print("3. Make non-decreasing")
    print("4. Can fix with 1 change?")
    print("5. Full analysis")

    choice = input("\nChoice (1-5): ")

    print("\n" + "=" * 45)

    if choice == '1':
        result = all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
        print(f"Array: {arr}")
        print(f"Non-decreasing: {'✅ Yes' if result else '❌ No'}")

    elif choice == '2':
        print(f"Array: {arr}")
        violations = [(i, arr[i], arr[i+1]) for i in range(len(arr)-1) if arr[i] > arr[i+1]]
        if violations:
            print(f"Violations found: {len(violations)}")
            for idx, a, b in violations:
                print(f"  Index {idx}: {a} > {b}")
        else:
            print("No violations! Array is non-decreasing.")

    elif choice == '3':
        modified = arr.copy()
        for i in range(1, len(modified)):
            if modified[i] < modified[i-1]:
                modified[i] = modified[i-1]
        print(f"Original: {arr}")
        print(f"Modified: {modified}")

    elif choice == '4':
        violations = sum(1 for i in range(len(arr)-1) if arr[i] > arr[i+1])
        can_fix = violations <= 1
        print(f"Array: {arr}")
        print(f"Violations: {violations}")
        print(f"Can fix with 1 change: {'✅ Yes' if can_fix else '❌ No'}")

    elif choice == '5':
        print(f"📊 FULL ANALYSIS")
        print(f"Array: {arr}")
        print(f"Length: {len(arr)}")

        is_nd = all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
        print(f"Is Non-decreasing: {'✅' if is_nd else '❌'}")

        violations = [(i, arr[i], arr[i+1]) for i in range(len(arr)-1) if arr[i] > arr[i+1]]
        print(f"Violations: {len(violations)}")

        if violations:
            for idx, a, b in violations:
                print(f"  - Index {idx}: {a} > {b}")

        if not is_nd:
            modified = arr.copy()
            for i in range(1, len(modified)):
                if modified[i] < modified[i-1]:
                    modified[i] = modified[i-1]
            print(f"Modified to non-decreasing: {modified}")

# Run
non_decreasing_analyzer()


# In[12]:


class NonDecreasingArray:
    """Complete utility for non-decreasing array operations"""

    def __init__(self, arr):
        self.original = arr.copy()
        self.arr = arr.copy()

    def reset(self):
        """Reset to original"""
        self.arr = self.original.copy()
        return self

    def is_non_decreasing(self):
        """Check if array is non-decreasing"""
        return all(self.arr[i] <= self.arr[i+1] for i in range(len(self.arr)-1))

    def is_strictly_increasing(self):
        """Check if array is strictly increasing"""
        return all(self.arr[i] < self.arr[i+1] for i in range(len(self.arr)-1))

    def count_violations(self):
        """Count number of violations"""
        return sum(1 for i in range(len(self.arr)-1) if self.arr[i] > self.arr[i+1])

    def get_violations(self):
        """Get all violation details"""
        return [
            {'index': i, 'values': (self.arr[i], self.arr[i+1])}
            for i in range(len(self.arr)-1) 
            if self.arr[i] > self.arr[i+1]
        ]

    def can_fix_with_one_change(self):
        """Check if can be fixed with at most one change"""
        return self.count_violations() <= 1

    def make_non_decreasing(self, strategy='increase'):
        """
        Make array non-decreasing
        strategy: 'increase' - increase smaller elements
                  'decrease' - decrease larger elements (when possible)
                  'optimal' - choose best option
        """
        self.arr = self.original.copy()
        changes = []

        for i in range(1, len(self.arr)):
            if self.arr[i] < self.arr[i - 1]:
                if strategy == 'increase':
                    old = self.arr[i]
                    self.arr[i] = self.arr[i - 1]
                    changes.append(('increase', i, old, self.arr[i]))

                elif strategy == 'decrease':
                    if i == 1 or self.arr[i] >= self.arr[i - 2]:
                        old = self.arr[i - 1]
                        self.arr[i - 1] = self.arr[i]
                        changes.append(('decrease', i-1, old, self.arr[i-1]))
                    else:
                        old = self.arr[i]
                        self.arr[i] = self.arr[i - 1]
                        changes.append(('increase', i, old, self.arr[i]))

                elif strategy == 'optimal':
                    can_decrease = (i == 1) or (self.arr[i] >= self.arr[i - 2])

                    if can_decrease:
                        old = self.arr[i - 1]
                        self.arr[i - 1] = self.arr[i]
                        changes.append(('decrease', i-1, old, self.arr[i-1]))
                    else:
                        old = self.arr[i]
                        self.arr[i] = self.arr[i - 1]
                        changes.append(('increase', i, old, self.arr[i]))

        return self.arr, changes

    def longest_non_decreasing_subsequence(self):
        """Find longest non-decreasing subsequence"""
        n = len(self.arr)
        if n == 0:
            return []

        dp = [1] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if self.arr[j] <= self.arr[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_idx = dp.index(max(dp))

        subseq = []
        idx = max_idx
        while idx != -1:
            subseq.append(self.arr[idx])
            idx = parent[idx]

        return subseq[::-1]

    def min_changes_required(self):
        """Minimum changes to make non-decreasing"""
        lnds = len(self.longest_non_decreasing_subsequence())
        return len(self.arr) - lnds

    def full_analysis(self):
        """Complete analysis report"""
        print("\n" + "╔" + "═" * 55 + "╗")
        print("║" + "📊 NON-DECREASING ARRAY ANALYSIS".center(55) + "║")
        print("╠" + "═" * 55 + "╣")

        print(f"║  📝 Array: {self.original}".ljust(56) + "║")
        print(f"║  📏 Length: {len(self.original)}".ljust(56) + "║")
        print("╠" + "═" * 55 + "╣")

        is_nd = self.is_non_decreasing()
        is_si = self.is_strictly_increasing()
        violations = self.count_violations()

        nd_status = "✅ Yes" if is_nd else "❌ No"
        si_status = "✅ Yes" if is_si else "❌ No"

        print(f"║  📈 Non-Decreasing: {nd_status}".ljust(56) + "║")
        print(f"║  📈 Strictly Increasing: {si_status}".ljust(56) + "║")
        print(f"║  ⚠️  Violations: {violations}".ljust(56) + "║")

        if violations > 0:
            print("╠" + "═" * 55 + "╣")
            print("║  📍 VIOLATION DETAILS:".ljust(56) + "║")
            for v in self.get_violations():
                print(f"║     Index {v['index']}: {v['values'][0]} > {v['values'][1]}".ljust(56) + "║")

        print("╠" + "═" * 55 + "╣")

        can_fix = self.can_fix_with_one_change()
        fix_status = "✅ Yes" if can_fix else "❌ No"
        print(f"║  🔧 Can fix with 1 change: {fix_status}".ljust(56) + "║")

        lnds = self.longest_non_decreasing_subsequence()
        print(f"║  📈 Longest ND Subsequence: {lnds}".ljust(56) + "║")
        print(f"║  📊 Min changes needed: {self.min_changes_required()}".ljust(56) + "║")

        if not is_nd:
            print("╠" + "═" * 55 + "╣")
            modified, changes = self.make_non_decreasing('optimal')
            print(f"║  ✅ Modified Array: {modified}".ljust(56) + "║")

        print("╚" + "═" * 55 + "╝")


# Usage
arr = [4, 2, 5, 3, 7, 1, 6]
analyzer = NonDecreasingArray(arr)
analyzer.full_analysis()


# In[13]:


arr = [1, 3, 2, 4, 3, 5]

# Check non-decreasing
is_nd = all(a <= b for a, b in zip(arr, arr[1:]))
print(f"Non-decreasing: {is_nd}")

# Count violations
violations = sum(a > b for a, b in zip(arr, arr[1:]))
print(f"Violations: {violations}")

# Make non-decreasing (simple)
from functools import reduce
nd_arr = reduce(lambda acc, x: acc + [max(acc[-1], x)] if acc else [x], arr, [])
print(f"Made non-decreasing: {nd_arr}")

# Check using sorted
is_nd_sorted = arr == sorted(arr)
print(f"Using sorted: {is_nd_sorted}")

# Lambda checker
check = lambda a: all(a[i] <= a[i+1] for i in range(len(a)-1))
print(f"Lambda check: {check(arr)}")

