#!/usr/bin/env python
# coding: utf-8

# In[1]:


def left_rotate_by_k(arr, k):
    """
    Left rotate array by k positions
    Time: O(n), Space: O(n)
    """
    n = len(arr)
    k = k % n  # Handle k > n

    return arr[k:] + arr[:k]

# Example
arr = [1, 2, 3, 4, 5, 6, 7]
k = 3

result = left_rotate_by_k(arr, k)

print(f"Original: {arr}")
print(f"Left Rotate by {k}: {result}")


# In[2]:


def right_rotate_by_k(arr, k):
    """
    Right rotate array by k positions
    Time: O(n), Space: O(n)
    """
    n = len(arr)
    k = k % n  # Handle k > n

    return arr[-k:] + arr[:-k]

# Example
arr = [1, 2, 3, 4, 5, 6, 7]
k = 3

result = right_rotate_by_k(arr, k)

print(f"Original: {arr}")
print(f"Right Rotate by {k}: {result}")


# In[3]:


def rotate_with_steps(arr, k, direction='left'):
    """
    Show rotation step by step
    """
    arr = arr.copy()
    n = len(arr)
    k = k % n

    print(f"🔄 {direction.upper()} ROTATION by {k}")
    print("=" * 40)
    print(f"Start: {arr}")
    print("-" * 40)

    for step in range(k):
        if direction == 'left':
            # Move first element to end
            first = arr.pop(0)
            arr.append(first)
            print(f"Step {step + 1}: Move {first} to end → {arr}")
        else:
            # Move last element to front
            last = arr.pop()
            arr.insert(0, last)
            print(f"Step {step + 1}: Move {last} to front → {arr}")

    print("-" * 40)
    print(f"Final: {arr}")
    return arr

# Example
arr = [1, 2, 3, 4, 5]
rotate_with_steps(arr, 3, 'left')
print()
rotate_with_steps(arr, 3, 'right')


# In[4]:


def reverse(arr, start, end):
    """Reverse array from start to end index"""
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1

def left_rotate_reversal(arr, k):
    """
    Left rotate using reversal algorithm
    Time: O(n), Space: O(1) - In-place!
    """
    n = len(arr)
    k = k % n

    if k == 0:
        return arr

    # Step 1: Reverse first k elements
    reverse(arr, 0, k - 1)

    # Step 2: Reverse remaining elements
    reverse(arr, k, n - 1)

    # Step 3: Reverse entire array
    reverse(arr, 0, n - 1)

    return arr

def right_rotate_reversal(arr, k):
    """
    Right rotate using reversal algorithm
    Time: O(n), Space: O(1) - In-place!
    """
    n = len(arr)
    k = k % n

    if k == 0:
        return arr

    # Step 1: Reverse entire array
    reverse(arr, 0, n - 1)

    # Step 2: Reverse first k elements
    reverse(arr, 0, k - 1)

    # Step 3: Reverse remaining elements
    reverse(arr, k, n - 1)

    return arr

# Examples
arr1 = [1, 2, 3, 4, 5, 6, 7]
arr2 = [1, 2, 3, 4, 5, 6, 7]

print(f"Original: [1, 2, 3, 4, 5, 6, 7]")
print(f"Left Rotate by 3: {left_rotate_reversal(arr1, 3)}")
print(f"Right Rotate by 3: {right_rotate_reversal(arr2, 3)}")


# In[5]:


def left_rotate_loop(arr, k):
    """Left rotate using loop"""
    n = len(arr)
    k = k % n

    result = [0] * n

    for i in range(n):
        # New position = (i - k + n) % n
        new_pos = (i - k + n) % n
        result[new_pos] = arr[i]

    return result

def right_rotate_loop(arr, k):
    """Right rotate using loop"""
    n = len(arr)
    k = k % n

    result = [0] * n

    for i in range(n):
        # New position = (i + k) % n
        new_pos = (i + k) % n
        result[new_pos] = arr[i]

    return result

# Example
arr = [1, 2, 3, 4, 5]
k = 2

print(f"Original: {arr}")
print(f"Left Rotate by {k}: {left_rotate_loop(arr, k)}")
print(f"Right Rotate by {k}: {right_rotate_loop(arr, k)}")


# In[6]:


from collections import deque

def rotate_deque(arr, k, direction='left'):
    """
    Rotate using deque
    deque.rotate() is O(k)
    """
    d = deque(arr)

    if direction == 'left':
        d.rotate(-k)  # Negative = left
    else:
        d.rotate(k)   # Positive = right

    return list(d)

# Example
arr = [1, 2, 3, 4, 5, 6, 7]
k = 3

print(f"Original: {arr}")
print(f"Left Rotate by {k}: {rotate_deque(arr, k, 'left')}")
print(f"Right Rotate by {k}: {rotate_deque(arr, k, 'right')}")


# In[7]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])
k = 3

# Left rotation (negative)
left_rotated = np.roll(arr, -k)

# Right rotation (positive)
right_rotated = np.roll(arr, k)

print(f"Original: {arr}")
print(f"Left Rotate by {k}: {left_rotated}")
print(f"Right Rotate by {k}: {right_rotated}")


# In[8]:


import math

def left_rotate_juggling(arr, k):
    """
    Juggling algorithm - efficient for large arrays
    Time: O(n), Space: O(1)
    """
    n = len(arr)
    k = k % n

    if k == 0:
        return arr

    # Number of sets = GCD(n, k)
    num_sets = math.gcd(n, k)

    for i in range(num_sets):
        temp = arr[i]
        j = i

        while True:
            d = (j + k) % n
            if d == i:
                break
            arr[j] = arr[d]
            j = d

        arr[j] = temp

    return arr

# Example
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
k = 3

print(f"Original: {arr}")
result = left_rotate_juggling(arr.copy(), k)
print(f"Left Rotate by {k}: {result}")


# In[9]:


def array_rotation_by_k():
    """Interactive array rotation by k"""

    print("🔄 ARRAY ROTATION BY K")
    print("=" * 40)

    # Get array input
    user_input = input("Enter array elements (space separated): ")
    arr = list(map(int, user_input.split()))

    # Get k value
    k = int(input("Enter k (rotation amount): "))

    # Get direction
    print("\nSelect direction:")
    print("1. Left Rotation")
    print("2. Right Rotation")
    direction = input("Choice (1/2): ")

    n = len(arr)
    k = k % n  # Normalize k

    # Perform rotation
    if direction == '1':
        result = arr[k:] + arr[:k]
        dir_name = "Left"
    else:
        result = arr[-k:] + arr[:-k]
        dir_name = "Right"

    # Display result
    print("\n" + "=" * 40)
    print(f"📊 Original Array: {arr}")
    print(f"📍 K Value: {k}")
    print(f"🔄 Direction: {dir_name}")
    print(f"✅ Result: {result}")
    print("=" * 40)

# Run
array_rotation_by_k()


# In[10]:


def rotate_by_k_safe(arr, k, direction='left'):
    """
    Safe rotation with all edge cases handled
    """

    # Edge case: Empty array
    if not arr:
        return [], "❌ Array is empty!"

    # Edge case: Single element
    if len(arr) == 1:
        return arr.copy(), "ℹ️ Single element, no rotation needed"

    # Edge case: k is 0
    if k == 0:
        return arr.copy(), "ℹ️ k=0, no rotation"

    # Edge case: Negative k
    if k < 0:
        k = abs(k)
        direction = 'right' if direction == 'left' else 'left'

    n = len(arr)
    k = k % n  # Normalize k

    # Edge case: k equals array length (or multiple)
    if k == 0:
        return arr.copy(), "ℹ️ k is multiple of array length, same as original"

    # Perform rotation
    if direction == 'left':
        result = arr[k:] + arr[:k]
    else:
        result = arr[-k:] + arr[:-k]

    return result, f"✅ {direction.title()} rotated by {k}"

# Test all edge cases
test_cases = [
    ([1, 2, 3, 4, 5], 2, 'left'),    # Normal case
    ([1, 2, 3, 4, 5], 2, 'right'),   # Normal right
    ([1, 2, 3, 4, 5], 7, 'left'),    # k > n
    ([1, 2, 3, 4, 5], 5, 'left'),    # k = n
    ([1, 2, 3, 4, 5], 10, 'left'),   # k = 2n
    ([1, 2, 3, 4, 5], 0, 'left'),    # k = 0
    ([1, 2, 3, 4, 5], -2, 'left'),   # Negative k
    ([42], 5, 'left'),               # Single element
    ([], 3, 'left'),                 # Empty array
]

print("🔄 ROTATION TEST CASES")
print("=" * 60)

for arr, k, direction in test_cases:
    result, message = rotate_by_k_safe(arr, k, direction)
    print(f"\nInput: arr={arr}, k={k}, dir={direction}")
    print(f"Output: {result}")
    print(f"Status: {message}")


# In[11]:


class ArrayRotator:
    """Complete array rotation utility"""

    def __init__(self, arr):
        self.original = arr.copy()
        self.current = arr.copy()

    def reset(self):
        """Reset to original"""
        self.current = self.original.copy()
        return self

    def left(self, k):
        """Left rotate by k"""
        n = len(self.current)
        k = k % n
        self.current = self.current[k:] + self.current[:k]
        return self

    def right(self, k):
        """Right rotate by k"""
        n = len(self.current)
        k = k % n
        self.current = self.current[-k:] + self.current[:-k]
        return self

    def get(self):
        """Get current array"""
        return self.current

    def show_all_rotations(self):
        """Display all possible rotations"""
        n = len(self.original)

        print(f"\n📊 ALL ROTATIONS OF {self.original}")
        print("=" * 55)
        print(f"{'K':<5}{'Left Rotation':<25}{'Right Rotation':<25}")
        print("-" * 55)

        for k in range(n):
            left = self.original[k:] + self.original[:k]
            right = self.original[-k:] + self.original[:-k] if k > 0 else self.original.copy()
            print(f"{k:<5}{str(left):<25}{str(right):<25}")

    def find_rotation_count(self, target):
        """Find how many rotations needed to get target"""
        n = len(self.original)

        for k in range(n):
            # Check left rotation
            left = self.original[k:] + self.original[:k]
            if left == target:
                return k, 'left'

            # Check right rotation
            right = self.original[-k:] + self.original[:-k] if k > 0 else self.original.copy()
            if right == target:
                return k, 'right'

        return None, None

    def chain_rotations(self, operations):
        """
        Apply multiple rotations
        operations = [('left', 2), ('right', 1), ...]
        """
        print(f"\n🔄 CHAIN ROTATIONS")
        print(f"Start: {self.current}")

        for direction, k in operations:
            if direction == 'left':
                self.left(k)
            else:
                self.right(k)
            print(f"After {direction} by {k}: {self.current}")

        return self.current

# Usage
arr = [1, 2, 3, 4, 5]
rotator = ArrayRotator(arr)

print("🔄 ARRAY ROTATOR")
print("=" * 40)
print(f"Original: {arr}")

# Chain methods
result = rotator.left(2).right(1).get()
print(f"\nLeft(2) then Right(1): {result}")

rotator.reset()
rotator.show_all_rotations()

# Find rotation
target = [4, 5, 1, 2, 3]
k, direction = rotator.find_rotation_count(target)
print(f"\nTo get {target}:")
print(f"Need {direction} rotation by {k}")

# Chain operations
rotator.reset()
operations = [('left', 2), ('right', 1), ('left', 3)]
rotator.chain_rotations(operations)


# In[12]:


import time
from collections import deque
import numpy as np

def measure_time(func, arr, k, iterations=1000):
    """Measure execution time"""
    start = time.time()
    for _ in range(iterations):
        func(arr.copy(), k)
    end = time.time()
    return (end - start) / iterations * 1000  # ms

# Methods
def slice_method(arr, k):
    k = k % len(arr)
    return arr[k:] + arr[:k]

def deque_method(arr, k):
    d = deque(arr)
    d.rotate(-k)
    return list(d)

def reversal_method(arr, k):
    n = len(arr)
    k = k % n

    def reverse(a, s, e):
        while s < e:
            a[s], a[e] = a[e], a[s]
            s += 1
            e -= 1

    reverse(arr, 0, k-1)
    reverse(arr, k, n-1)
    reverse(arr, 0, n-1)
    return arr

def loop_method(arr, k):
    n = len(arr)
    k = k % n
    result = [0] * n
    for i in range(n):
        result[(i - k + n) % n] = arr[i]
    return result

# Test
arr = list(range(1000))
k = 100

print("⚡ PERFORMANCE COMPARISON")
print("=" * 45)
print(f"Array size: {len(arr)}, K: {k}")
print("-" * 45)

methods = [
    ("Slicing", slice_method),
    ("Deque", deque_method),
    ("Reversal (in-place)", reversal_method),
    ("Loop", loop_method),
]

for name, func in methods:
    time_ms = measure_time(func, arr, k)
    print(f"{name:<20}: {time_ms:.4f} ms")

print("-" * 45)
print("✅ Slicing & Reversal are fastest!")


# In[13]:


arr = [1, 2, 3, 4, 5, 6, 7]
k = 3

# Left rotation
left = arr[k:] + arr[:k]
print(f"Left by {k}: {left}")

# Right rotation
right = arr[-k:] + arr[:-k]
print(f"Right by {k}: {right}")

# Using deque
from collections import deque
d = deque(arr); d.rotate(-k); print(f"Deque: {list(d)}")

# Using numpy
import numpy as np
print(f"NumPy: {list(np.roll(arr, -k))}")

# Lambda
rotate = lambda a, k: a[k%len(a):] + a[:k%len(a)]
print(f"Lambda: {rotate(arr, k)}")

# List comprehension
rotated = [arr[(i + k) % len(arr)] for i in range(len(arr))]
print(f"Comprehension: {rotated}")

