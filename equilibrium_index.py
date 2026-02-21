#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def equilibrium_index_brute(arr):
    """
    Find equilibrium index using brute force
    Time: O(n²), Space: O(1)
    """
    n = len(arr)

    for i in range(n):
        # Calculate left sum
        left_sum = sum(arr[:i])

        # Calculate right sum
        right_sum = sum(arr[i+1:])

        if left_sum == right_sum:
            return i

    return -1  # No equilibrium index found

# Example
arr = [-7, 1, 5, 2, -4, 3, 0]
result = equilibrium_index_brute(arr)

print(f"Array: {arr}")
print(f"Equilibrium Index: {result}")

# Verify
if result != -1:
    left = sum(arr[:result])
    right = sum(arr[result+1:])
    print(f"Left Sum: {left}")
    print(f"Right Sum: {right}")


# In[ ]:


def equilibrium_index_optimized(arr):
    """
    Find equilibrium index in single pass
    Time: O(n), Space: O(1)
    """
    n = len(arr)

    if n == 0:
        return -1

    # Calculate total sum
    total_sum = sum(arr)

    left_sum = 0

    for i in range(n):
        # Right sum = total - left - current element
        right_sum = total_sum - left_sum - arr[i]

        if left_sum == right_sum:
            return i

        # Add current element to left sum for next iteration
        left_sum += arr[i]

    return -1

# Example
arr = [-7, 1, 5, 2, -4, 3, 0]
result = equilibrium_index_optimized(arr)

print(f"Array: {arr}")
print(f"Equilibrium Index: {result}")


# In[ ]:


def equilibrium_with_visualization(arr):
    """
    Find equilibrium index with step-by-step visualization
    """
    n = len(arr)

    if n == 0:
        print("❌ Empty array!")
        return -1

    total_sum = sum(arr)

    print(f"🔍 FINDING EQUILIBRIUM INDEX")
    print("=" * 55)
    print(f"Array: {arr}")
    print(f"Total Sum: {total_sum}")
    print("-" * 55)
    print(f"{'Index':<7}{'Element':<10}{'Left Sum':<12}{'Right Sum':<12}{'Match'}")
    print("-" * 55)

    left_sum = 0
    equilibrium = -1

    for i in range(n):
        right_sum = total_sum - left_sum - arr[i]

        match = "✅ YES!" if left_sum == right_sum else "❌"

        print(f"{i:<7}{arr[i]:<10}{left_sum:<12}{right_sum:<12}{match}")

        if left_sum == right_sum and equilibrium == -1:
            equilibrium = i

        left_sum += arr[i]

    print("-" * 55)

    if equilibrium != -1:
        print(f"✅ Equilibrium Index Found: {equilibrium}")
        print(f"   Element at index {equilibrium}: {arr[equilibrium]}")
    else:
        print("❌ No Equilibrium Index Found!")

    return equilibrium

# Example
arr = [-7, 1, 5, 2, -4, 3, 0]
equilibrium_with_visualization(arr)


# In[ ]:


def find_all_equilibrium_indices(arr):
    """
    Find all equilibrium indices in array
    Time: O(n), Space: O(k) where k = number of equilibrium indices
    """
    n = len(arr)

    if n == 0:
        return []

    total_sum = sum(arr)
    left_sum = 0
    equilibrium_indices = []

    for i in range(n):
        right_sum = total_sum - left_sum - arr[i]

        if left_sum == right_sum:
            equilibrium_indices.append(i)

        left_sum += arr[i]

    return equilibrium_indices

# Example with multiple equilibrium indices
arr = [0, -3, 5, -4, -2, 3, 1, 0]

result = find_all_equilibrium_indices(arr)

print(f"Array: {arr}")
print(f"All Equilibrium Indices: {result}")
print()

# Verify each
for idx in result:
    left = sum(arr[:idx])
    right = sum(arr[idx+1:])
    print(f"Index {idx}: Left Sum = {left}, Right Sum = {right}")


# In[ ]:


def equilibrium_prefix_sum(arr):
    """
    Find equilibrium index using prefix sum
    Time: O(n), Space: O(n)
    """
    n = len(arr)

    if n == 0:
        return -1

    # Build prefix sum array
    prefix = [0] * n
    prefix[0] = arr[0]

    for i in range(1, n):
        prefix[i] = prefix[i - 1] + arr[i]

    total = prefix[-1]

    # Find equilibrium
    for i in range(n):
        left_sum = prefix[i] - arr[i]  # Sum before index i
        right_sum = total - prefix[i]   # Sum after index i

        if left_sum == right_sum:
            return i

    return -1

# Example
arr = [-7, 1, 5, 2, -4, 3, 0]

print(f"Array: {arr}")

# Calculate prefix sum
prefix = []
total = 0
for x in arr:
    total += x
    prefix.append(total)

print(f"Prefix Sum: {prefix}")
print(f"Equilibrium Index: {equilibrium_prefix_sum(arr)}")


# In[ ]:


def equilibrium_index_safe(arr):
    """
    Find equilibrium index with all edge cases handled
    """
    # Edge case: Empty array
    if not arr:
        return -1, "Array is empty"

    # Edge case: Single element
    if len(arr) == 1:
        return 0, "Single element - always equilibrium"

    # Edge case: Two elements
    if len(arr) == 2:
        if arr[0] == 0:
            return 0, "First element is 0"
        if arr[1] == 0:
            return 1, "Last element is 0"
        return -1, "No equilibrium for 2 non-zero elements"

    # Normal case
    total_sum = sum(arr)
    left_sum = 0

    for i in range(len(arr)):
        right_sum = total_sum - left_sum - arr[i]

        if left_sum == right_sum:
            return i, "Equilibrium found"

        left_sum += arr[i]

    return -1, "No equilibrium index exists"

# Test all edge cases
test_cases = [
    [],                          # Empty
    [5],                         # Single element
    [1, 1],                      # Two equal elements
    [0, 5],                      # Zero at start
    [5, 0],                      # Zero at end
    [1, 2, 3],                   # No equilibrium
    [-7, 1, 5, 2, -4, 3, 0],    # Normal case
    [1, 2, 3, 4, 6],             # No equilibrium
    [0, 0, 0, 0],                # All zeros
    [1, -1, 1, -1, 1],           # Alternating
]

print("🔍 EQUILIBRIUM INDEX - EDGE CASES")
print("=" * 60)

for arr in test_cases:
    idx, msg = equilibrium_index_safe(arr)

    if idx != -1:
        print(f"{str(arr):<30} → Index {idx} ({msg})")
    else:
        print(f"{str(arr):<30} → ❌ {msg}")


# In[ ]:


def equilibrium_analyzer():
    """Interactive equilibrium index finder"""

    print("⚖️ EQUILIBRIUM INDEX FINDER")
    print("=" * 45)

    # Get input
    user_input = input("Enter array elements (space separated): ")
    arr = list(map(int, user_input.split()))

    print("\n📋 MENU")
    print("1. Find first equilibrium index")
    print("2. Find all equilibrium indices")
    print("3. Step-by-step visualization")
    print("4. Full analysis")

    choice = input("\nChoice (1-4): ")

    print("\n" + "=" * 45)

    if choice == '1':
        total = sum(arr)
        left = 0
        result = -1

        for i in range(len(arr)):
            right = total - left - arr[i]
            if left == right:
                result = i
                break
            left += arr[i]

        print(f"Array: {arr}")
        if result != -1:
            print(f"✅ First Equilibrium Index: {result}")
            print(f"   Element: {arr[result]}")
            print(f"   Left Sum: {sum(arr[:result])}")
            print(f"   Right Sum: {sum(arr[result+1:])}")
        else:
            print("❌ No equilibrium index found!")

    elif choice == '2':
        total = sum(arr)
        left = 0
        indices = []

        for i in range(len(arr)):
            right = total - left - arr[i]
            if left == right:
                indices.append(i)
            left += arr[i]

        print(f"Array: {arr}")
        print(f"Equilibrium Indices: {indices if indices else 'None'}")

    elif choice == '3':
        equilibrium_with_visualization(arr)

    elif choice == '4':
        total = sum(arr)
        left = 0
        indices = []

        for i in range(len(arr)):
            right = total - left - arr[i]
            if left == right:
                indices.append(i)
            left += arr[i]

        print(f"📊 FULL ANALYSIS")
        print(f"Array: {arr}")
        print(f"Length: {len(arr)}")
        print(f"Total Sum: {total}")
        print(f"Equilibrium Indices: {indices if indices else 'None'}")
        print(f"Count: {len(indices)}")

        if indices:
            print("\n📍 Details:")
            for idx in indices:
                print(f"   Index {idx}: arr[{idx}] = {arr[idx]}")
                print(f"      Left Sum:  {sum(arr[:idx])}")
                print(f"      Right Sum: {sum(arr[idx+1:])}")

# Run
equilibrium_analyzer()


# In[ ]:


import numpy as np

def equilibrium_numpy(arr):
    """
    Find equilibrium indices using NumPy
    """
    arr = np.array(arr)
    n = len(arr)

    if n == 0:
        return []

    # Calculate cumulative sum
    cumsum = np.cumsum(arr)
    total = cumsum[-1]

    # Left sum = cumsum - current element
    left_sum = np.concatenate([[0], cumsum[:-1]])

    # Right sum = total - cumsum
    right_sum = total - cumsum

    # Find where left == right
    equilibrium_mask = (left_sum == right_sum)
    equilibrium_indices = np.where(equilibrium_mask)[0]

    return equilibrium_indices.tolist()

# Example
arr = [-7, 1, 5, 2, -4, 3, 0]

result = equilibrium_numpy(arr)

print(f"Array: {arr}")
print(f"Equilibrium Indices (NumPy): {result}")

# Show cumulative sums
cumsum = np.cumsum(arr)
print(f"\nCumulative Sum: {cumsum.tolist()}")


# In[ ]:


class EquilibriumFinder:
    """Complete utility for equilibrium index operations"""

    def __init__(self, arr):
        self.arr = arr.copy()
        self.n = len(arr)

    def find_first(self):
        """Find first equilibrium index"""
        if self.n == 0:
            return -1

        total = sum(self.arr)
        left = 0

        for i in range(self.n):
            right = total - left - self.arr[i]
            if left == right:
                return i
            left += self.arr[i]

        return -1

    def find_all(self):
        """Find all equilibrium indices"""
        if self.n == 0:
            return []

        total = sum(self.arr)
        left = 0
        indices = []

        for i in range(self.n):
            right = total - left - self.arr[i]
            if left == right:
                indices.append(i)
            left += self.arr[i]

        return indices

    def count(self):
        """Count equilibrium indices"""
        return len(self.find_all())

    def has_equilibrium(self):
        """Check if any equilibrium exists"""
        return self.find_first() != -1

    def verify_index(self, idx):
        """Verify if given index is equilibrium"""
        if idx < 0 or idx >= self.n:
            return False, "Index out of bounds"

        left_sum = sum(self.arr[:idx])
        right_sum = sum(self.arr[idx + 1:])

        is_eq = left_sum == right_sum

        return is_eq, {
            'index': idx,
            'element': self.arr[idx],
            'left_sum': left_sum,
            'right_sum': right_sum
        }

    def get_details(self, idx):
        """Get detailed info for an equilibrium index"""
        if idx < 0 or idx >= self.n:
            return None

        left_part = self.arr[:idx]
        right_part = self.arr[idx + 1:]

        return {
            'index': idx,
            'element': self.arr[idx],
            'left_part': left_part,
            'right_part': right_part,
            'left_sum': sum(left_part),
            'right_sum': sum(right_part)
        }

    def visualize(self, idx):
        """Visualize equilibrium at given index"""
        if idx < 0 or idx >= self.n:
            print("❌ Invalid index!")
            return

        details = self.get_details(idx)

        left_str = " + ".join(map(str, details['left_part'])) if details['left_part'] else "0"
        right_str = " + ".join(map(str, details['right_part'])) if details['right_part'] else "0"

        print(f"\n⚖️ EQUILIBRIUM VISUALIZATION")
        print("=" * 50)
        print(f"Array: {self.arr}")
        print(f"Index: {idx}, Element: [{details['element']}]")
        print()
        print(f"  Left Side              ⚖️              Right Side")
        print(f"  {left_str}")
        print(f"     = {details['left_sum']}                              = {details['right_sum']}")
        print(f"                         [{details['element']}]")
        print()

        if details['left_sum'] == details['right_sum']:
            print(f"  ✅ BALANCED! ({details['left_sum']} = {details['right_sum']})")
        else:
            print(f"  ❌ NOT BALANCED! ({details['left_sum']} ≠ {details['right_sum']})")

    def full_analysis(self):
        """Complete analysis report"""
        print("\n" + "╔" + "═" * 55 + "╗")
        print("║" + "⚖️ EQUILIBRIUM INDEX ANALYSIS".center(55) + "║")
        print("╠" + "═" * 55 + "╣")

        print(f"║  📝 Array: {self.arr}".ljust(56) + "║")
        print(f"║  📏 Length: {self.n}".ljust(56) + "║")
        print(f"║  📊 Total Sum: {sum(self.arr)}".ljust(56) + "║")
        print("╠" + "═" * 55 + "╣")

        indices = self.find_all()
        has_eq = len(indices) > 0

        print(f"║  ⚖️ Has Equilibrium: {'✅ Yes' if has_eq else '❌ No'}".ljust(56) + "║")
        print(f"║  📍 Equilibrium Count: {len(indices)}".ljust(56) + "║")

        if indices:
            print(f"║  📋 Indices: {indices}".ljust(56) + "║")
            print("╠" + "═" * 55 + "╣")
            print("║  📊 EQUILIBRIUM DETAILS:".ljust(56) + "║")

            for idx in indices:
                details = self.get_details(idx)
                print(f"║  ".ljust(56) + "║")
                print(f"║  Index {idx}: Element = {details['element']}".ljust(56) + "║")
                print(f"║     Left:  {details['left_part']} = {details['left_sum']}".ljust(56) + "║")
                print(f"║     Right: {details['right_part']} = {details['right_sum']}".ljust(56) + "║")

        print("╚" + "═" * 55 + "╝")


# Usage
arr = [-7, 1, 5, 2, -4, 3, 0]
finder = EquilibriumFinder(arr)

# Full analysis
finder.full_analysis()

# Visualize specific index
finder.visualize(3)


# In[ ]:


def pivot_index(arr):
    """
    Find pivot index - same as equilibrium
    LeetCode Problem 724
    """
    total = sum(arr)
    left_sum = 0

    for i in range(len(arr)):
        # Check if left sum equals right sum
        if left_sum == total - left_sum - arr[i]:
            return i
        left_sum += arr[i]

    return -1

# Example
arr = [1, 7, 3, 6, 5, 6]
print(f"Array: {arr}")
print(f"Pivot Index: {pivot_index(arr)}")  # Output: 3
# Left: 1+7+3 = 11, Right: 5+6 = 11


# In[ ]:


def can_split_equal_sum(arr):
    """
    Check if array can be split into two parts with equal sum
    """
    total = sum(arr)

    # If total is odd, can't split equally
    if total % 2 != 0:
        return False, -1

    target = total // 2
    current_sum = 0

    for i in range(len(arr) - 1):
        current_sum += arr[i]
        if current_sum == target:
            return True, i

    return False, -1

# Example
arr = [1, 2, 3, 4, 5, 5]
can_split, split_index = can_split_equal_sum(arr)

print(f"Array: {arr}")
print(f"Can Split: {can_split}")
if can_split:
    print(f"Split after index: {split_index}")
    print(f"Left part: {arr[:split_index+1]} = {sum(arr[:split_index+1])}")
    print(f"Right part: {arr[split_index+1:]} = {sum(arr[split_index+1:])}")


# In[ ]:


import time

def measure_time(func, arr, iterations=1000):
    """Measure execution time"""
    start = time.time()
    for _ in range(iterations):
        func(arr.copy())
    end = time.time()
    return (end - start) / iterations * 1000

# Different methods
def brute_force(arr):
    for i in range(len(arr)):
        if sum(arr[:i]) == sum(arr[i+1:]):
            return i
    return -1

def optimized(arr):
    total = sum(arr)
    left = 0
    for i in range(len(arr)):
        if left == total - left - arr[i]:
            return i
        left += arr[i]
    return -1

def prefix_sum(arr):
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
    total = prefix[-1]
    for i in range(n):
        if prefix[i] == total - prefix[i + 1]:
            return i
    return -1

# Test
arr = list(range(100))  # Array of size 100

print("⚡ PERFORMANCE COMPARISON")
print("=" * 40)
print(f"Array size: {len(arr)}")
print("-" * 40)

methods = [
    ("Brute Force O(n²)", brute_force),
    ("Optimized O(n)", optimized),
    ("Prefix Sum O(n)", prefix_sum),
]

for name, func in methods:
    time_ms = measure_time(func, arr)
    print(f"{name:<20}: {time_ms:.4f} ms")

print("-" * 40)
print("✅ Optimized method is fastest!")


# In[ ]:


arr = [-7, 1, 5, 2, -4, 3, 0]

# Find first equilibrium
eq = next((i for i in range(len(arr)) if sum(arr[:i]) == sum(arr[i+1:])), -1)
print(f"First Equilibrium: {eq}")

# Find all equilibrium indices
all_eq = [i for i in range(len(arr)) if sum(arr[:i]) == sum(arr[i+1:])]
print(f"All Equilibrium: {all_eq}")

# Check if equilibrium exists
has_eq = any(sum(arr[:i]) == sum(arr[i+1:]) for i in range(len(arr)))
print(f"Has Equilibrium: {has_eq}")

# Using accumulate
from itertools import accumulate
total = sum(arr)
prefix = list(accumulate(arr, initial=0))
eq_indices = [i for i in range(len(arr)) if prefix[i] == total - prefix[i+1]]
print(f"Using accumulate: {eq_indices}")

