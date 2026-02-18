#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Method 1: Sort and get second last element
numbers = [12, 35, 1, 10, 34, 1]

# Sort the list
sorted_list = sorted(numbers, reverse=True)
second_largest = sorted_list[1]

print(f"List: {numbers}")
print(f"Second Largest: {second_largest}")


# In[ ]:


# Handle duplicates properly
numbers = [10, 20, 20, 20, 10, 5]

# Remove duplicates using set, then sort
unique_sorted = sorted(set(numbers), reverse=True)
second_largest = unique_sorted[1]

print(f"List: {numbers}")
print(f"Unique Sorted: {unique_sorted}")
print(f"Second Largest: {second_largest}")


# In[ ]:


def second_largest(numbers):
    """
    Find second largest in single traversal
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(numbers) < 2:
        return None

    first = second = float('-inf')

    for num in numbers:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num

    if second == float('-inf'):
        return None

    return second

# Example
numbers = [12, 35, 1, 10, 34, 1]
result = second_largest(numbers)

print(f"List: {numbers}")
print(f"Largest: {max(numbers)}")
print(f"Second Largest: {result}")


# In[ ]:


def find_second_largest(numbers):
    """Find second largest with detailed steps"""

    if len(numbers) < 2:
        print("❌ Need at least 2 elements!")
        return None

    # Initialize
    largest = numbers[0]
    second_largest = float('-inf')

    print(f"📝 Initial: largest = {largest}")
    print("-" * 40)

    for i, num in enumerate(numbers[1:], 1):
        print(f"Step {i}: Checking {num}")

        if num > largest:
            second_largest = largest
            largest = num
            print(f"   → New largest: {largest}")
            print(f"   → Second largest updated to: {second_largest}")
        elif num > second_largest and num != largest:
            second_largest = num
            print(f"   → Second largest updated to: {second_largest}")
        else:
            print(f"   → No change")

    print("-" * 40)
    return second_largest

# Example
numbers = [12, 35, 1, 10, 34, 1]
result = find_second_largest(numbers)
print(f"\n✅ Second Largest: {result}")


# In[ ]:


# Method using max() function
numbers = [12, 35, 1, 10, 34, 1]

# Create a copy to avoid modifying original
temp_list = numbers.copy()

# Remove the largest
largest = max(temp_list)
temp_list.remove(largest)

# Get second largest
second_largest = max(temp_list)

print(f"List: {numbers}")
print(f"Largest: {largest}")
print(f"Second Largest: {second_largest}")


# In[ ]:


import heapq

# Method 1: nlargest
numbers = [12, 35, 1, 10, 34, 1]

# Get 2 largest elements
two_largest = heapq.nlargest(2, numbers)
second_largest = two_largest[1]

print(f"List: {numbers}")
print(f"Two Largest: {two_largest}")
print(f"Second Largest: {second_largest}")

# Method 2: With unique values
unique_numbers = list(set(numbers))
two_largest_unique = heapq.nlargest(2, unique_numbers)
print(f"\nUnique Second Largest: {two_largest_unique[1]}")


# In[ ]:


import numpy as np

numbers = [12, 35, 1, 10, 34, 1]

# Convert to numpy array
arr = np.array(numbers)

# Method 1: Using partition
# Partition puts nth largest at position n from end
partitioned = np.partition(arr, -2)
second_largest = partitioned[-2]

print(f"List: {numbers}")
print(f"Second Largest: {second_largest}")

# Method 2: Using argsort
sorted_indices = np.argsort(arr)
second_largest_idx = sorted_indices[-2]
print(f"Second Largest (argsort): {arr[second_largest_idx]}")

# Method 3: Unique values
unique = np.unique(arr)
second_largest_unique = unique[-2]
print(f"Second Largest (unique): {second_largest_unique}")


# In[ ]:


def get_second_largest():
    """Interactive second largest finder"""

    print("🔢 SECOND LARGEST ELEMENT FINDER")
    print("=" * 35)

    # Get input
    user_input = input("Enter numbers (space separated): ")
    numbers = list(map(int, user_input.split()))

    if len(numbers) < 2:
        print("❌ Need at least 2 numbers!")
        return

    # Find second largest
    first = second = float('-inf')

    for num in numbers:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num

    # Display result
    print("\n" + "=" * 35)
    print(f"📊 List: {numbers}")
    print(f"📈 Largest: {first}")
    print(f"📉 Second Largest: {second}")
    print("=" * 35)

# Run
get_second_largest()


# In[ ]:


def second_largest_complete(numbers):
    """
    Find second largest with all edge cases handled
    """

    # Edge case: Empty list
    if not numbers:
        return None, "List is empty!"

    # Edge case: Single element
    if len(numbers) == 1:
        return None, "Need at least 2 elements!"

    # Edge case: All same elements
    unique = set(numbers)
    if len(unique) == 1:
        return None, "All elements are same!"

    # Find second largest
    first = second = float('-inf')

    for num in numbers:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num

    return second, "Success"

# Test cases
test_cases = [
    [12, 35, 1, 10, 34, 1],  # Normal case
    [5, 5, 5, 5],             # All same
    [10],                     # Single element
    [],                       # Empty
    [1, 2],                   # Two elements
    [-5, -2, -10, -1],        # Negative numbers
    [100, 100, 99, 99, 98],   # Duplicates
]

print("🔢 SECOND LARGEST - TEST CASES")
print("=" * 50)

for i, test in enumerate(test_cases, 1):
    result, message = second_largest_complete(test)
    print(f"\nTest {i}: {test}")
    if result is not None:
        print(f"   ✅ Second Largest: {result}")
    else:
        print(f"   ❌ {message}")


# In[ ]:


def nth_largest(numbers, n):
    """Find nth largest element"""

    if not numbers or n < 1:
        return None

    unique_sorted = sorted(set(numbers), reverse=True)

    if n > len(unique_sorted):
        return None

    return unique_sorted[n - 1]

# Examples
numbers = [12, 35, 1, 10, 34, 1, 35, 10]

print(f"List: {numbers}")
print(f"Unique Sorted: {sorted(set(numbers), reverse=True)}")
print()

for i in range(1, 6):
    result = nth_largest(numbers, i)
    ordinal = ['1st', '2nd', '3rd', '4th', '5th'][i-1]
    print(f"{ordinal} Largest: {result}")


# In[ ]:


def second_largest_with_index(numbers):
    """Find second largest and its index"""

    if len(numbers) < 2:
        return None, None

    # Find largest
    largest = max(numbers)
    largest_idx = numbers.index(largest)

    # Find second largest (excluding largest value)
    second = float('-inf')
    second_idx = -1

    for i, num in enumerate(numbers):
        if num > second and num < largest:
            second = num
            second_idx = i
        elif num == largest and i != largest_idx and num > second:
            # Handle case when second largest equals largest
            second = num
            second_idx = i

    if second == float('-inf'):
        return None, None

    return second, second_idx

# Example
numbers = [12, 35, 1, 10, 34, 1]

value, index = second_largest_with_index(numbers)

print(f"List: {numbers}")
print(f"Indices: {list(range(len(numbers)))}")
print(f"\n✅ Second Largest: {value}")
print(f"📍 Index: {index}")


# In[ ]:


numbers = [12, 35, 1, 10, 34, 1]

# Method 1: Using sorted
second = sorted(numbers)[-2]
print(f"Method 1: {second}")

# Method 2: Using sorted with set (unique)
second_unique = sorted(set(numbers))[-2]
print(f"Method 2 (unique): {second_unique}")

# Method 3: Using heapq
import heapq
second_heap = heapq.nlargest(2, numbers)[1]
print(f"Method 3 (heapq): {second_heap}")

# Method 4: Using max and filter
largest = max(numbers)
second_filter = max(x for x in numbers if x != largest)
print(f"Method 4 (filter): {second_filter}")

# Method 5: Lambda with sorted
second_lambda = (lambda x: sorted(set(x))[-2])(numbers)
print(f"Method 5 (lambda): {second_lambda}")

