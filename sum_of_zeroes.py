#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def count_zeroes_in_number(n):
    """
    Count number of zeroes in a number
    Time: O(d), where d = number of digits
    """
    n = abs(n)  # Handle negative
    count = 0

    # Special case: n is 0
    if n == 0:
        return 1

    while n > 0:
        if n % 10 == 0:
            count += 1
        n //= 10

    return count

# Alternative: Using string
def count_zeroes_string(n):
    return str(abs(n)).count('0')

# Examples
test_numbers = [10203, 1000, 12345, 0, 100000, 505050]

print("🔢 COUNT ZEROES IN NUMBER")
print("=" * 40)

for n in test_numbers:
    count1 = count_zeroes_in_number(n)
    count2 = count_zeroes_string(n)
    print(f"{n} → {count1} zeroes")


# In[ ]:


def count_zeroes_in_array(arr):
    """
    Count number of zeroes in array
    Time: O(n)
    """
    count = 0

    for num in arr:
        if num == 0:
            count += 1

    return count

# One-liner alternatives
def count_zeroes_v2(arr):
    return arr.count(0)

def count_zeroes_v3(arr):
    return sum(1 for x in arr if x == 0)

# Examples
test_arrays = [
    [1, 0, 2, 0, 3, 0],
    [0, 0, 0, 0],
    [1, 2, 3, 4, 5],
    [0],
    [],
    [1, 0, -1, 0, 2, 0, -2],
]

print("🔢 COUNT ZEROES IN ARRAY")
print("=" * 50)

for arr in test_arrays:
    count = count_zeroes_in_array(arr)
    print(f"{arr} → {count} zeroes")


# In[ ]:


def sum_of_zero_indices(arr):
    """
    Find sum of indices where zeroes are present
    Time: O(n)
    """
    index_sum = 0
    zero_indices = []

    for i, num in enumerate(arr):
        if num == 0:
            index_sum += i
            zero_indices.append(i)

    return index_sum, zero_indices

# Examples
test_arrays = [
    [1, 0, 2, 0, 3],      # indices 1, 3 → sum = 4
    [0, 1, 0, 2, 0],      # indices 0, 2, 4 → sum = 6
    [0, 0, 0, 0],         # indices 0, 1, 2, 3 → sum = 6
    [1, 2, 3, 4, 5],      # no zeroes → sum = 0
]

print("🔢 SUM OF ZERO INDICES")
print("=" * 55)

for arr in test_arrays:
    total, indices = sum_of_zero_indices(arr)
    print(f"Array: {arr}")
    print(f"   Zero indices: {indices}")
    print(f"   Sum of indices: {total}")
    print()


# In[ ]:


def trailing_zeroes_factorial(n):
    """
    Count trailing zeroes in n!
    Trailing zeroes come from 10 = 2 × 5
    Since 2s are more abundant, count 5s

    Time: O(log n)
    """
    count = 0
    power_of_5 = 5

    while power_of_5 <= n:
        count += n // power_of_5
        power_of_5 *= 5

    return count

# Alternative
def trailing_zeroes_v2(n):
    count = 0
    while n >= 5:
        n //= 5
        count += n
    return count

# Examples
test_numbers = [5, 10, 25, 100, 1000]

print("🔢 TRAILING ZEROES IN FACTORIAL")
print("=" * 50)

for n in test_numbers:
    zeroes = trailing_zeroes_factorial(n)
    print(f"{n}! has {zeroes} trailing zeroes")

# Verification for small number
import math
n = 10
fact = math.factorial(n)
print(f"\nVerification: {n}! = {fact}")
print(f"Trailing zeroes: {trailing_zeroes_factorial(n)}")


# In[ ]:


def count_zeroes_binary(n):
    """
    Count zeroes in binary representation
    """
    if n == 0:
        return 1

    binary = bin(abs(n))[2:]  # Remove '0b' prefix
    return binary.count('0')

def count_zeroes_binary_manual(n):
    """Manual counting without string conversion"""
    if n == 0:
        return 1

    n = abs(n)
    count = 0

    while n > 0:
        if n & 1 == 0:  # Check last bit
            count += 1
        n >>= 1  # Right shift

    return count

# Examples
test_numbers = [10, 15, 8, 7, 100, 0]

print("🔢 ZEROES IN BINARY")
print("=" * 50)

for n in test_numbers:
    binary = bin(n)[2:]
    zeroes = count_zeroes_binary(n)
    ones = binary.count('1')

    print(f"{n} = {binary}")
    print(f"   Zeroes: {zeroes}, Ones: {ones}")
    print()


# In[ ]:


def count_zeroes_matrix(matrix):
    """
    Count zeroes in 2D matrix
    """
    count = 0
    positions = []

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                count += 1
                positions.append((i, j))

    return count, positions

# One-liner
def count_zeroes_matrix_v2(matrix):
    return sum(row.count(0) for row in matrix)

# Example
matrix = [
    [1, 0, 2, 0],
    [0, 3, 0, 4],
    [5, 6, 0, 7],
    [0, 0, 8, 9]
]

count, positions = count_zeroes_matrix(matrix)

print("🔢 ZEROES IN MATRIX")
print("=" * 40)
print("Matrix:")
for row in matrix:
    print(f"   {row}")

print(f"\nZero count: {count}")
print(f"Zero positions: {positions}")


# In[ ]:


def count_zeroes_sorted(arr):
    """
    Count zeroes in sorted array using binary search
    Array contains only 0s and 1s, sorted
    Example: [0, 0, 0, 1, 1, 1, 1]

    Time: O(log n)
    """
    if not arr:
        return 0

    # Find first occurrence of 1
    left, right = 0, len(arr) - 1
    first_one = len(arr)  # Default: no 1s found

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == 1:
            first_one = mid
            right = mid - 1
        else:
            left = mid + 1

    return first_one  # All elements before first 1 are 0s

# Examples
test_arrays = [
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 0, 0, 1],
]

print("🔢 COUNT ZEROES IN SORTED BINARY ARRAY")
print("=" * 50)

for arr in test_arrays:
    count = count_zeroes_sorted(arr)
    print(f"{arr} → {count} zeroes")


# In[ ]:


def zeroes_analysis_visualized(arr):
    """
    Complete visualization of zeroes analysis
    """
    print("🔢 ZEROES ANALYSIS VISUALIZATION")
    print("=" * 55)
    print(f"Array: {arr}")
    print("-" * 55)

    # Find all zeroes
    zero_indices = []

    print("\n📊 SCANNING ARRAY:")
    for i, num in enumerate(arr):
        status = "🔴 ZERO!" if num == 0 else "⚪"
        print(f"   Index {i}: {num} {status}")

        if num == 0:
            zero_indices.append(i)

    # Statistics
    count = len(zero_indices)
    index_sum = sum(zero_indices)

    print("\n" + "-" * 55)
    print("📊 STATISTICS:")
    print(f"   Total elements: {len(arr)}")
    print(f"   Zero count: {count}")
    print(f"   Zero indices: {zero_indices}")
    print(f"   Sum of indices: {index_sum}")
    print(f"   Zero percentage: {count/len(arr)*100:.1f}%" if arr else "N/A")

    # Visualization
    if arr:
        print("\n📊 VISUAL:")
        visual = ""
        for num in arr:
            visual += "⬛" if num == 0 else "⬜"
        print(f"   {visual}")
        print(f"   (⬛ = Zero, ⬜ = Non-zero)")

    return count, zero_indices, index_sum

# Example
arr = [3, 0, 1, 0, 0, 2, 0, 5, 0, 1]
zeroes_analysis_visualized(arr)


# In[ ]:


def move_zeroes_to_end(arr):
    """
    Move all zeroes to end, maintain order of other elements
    Time: O(n), Space: O(1)
    """
    arr = arr.copy()
    non_zero_idx = 0

    # Move non-zero elements to front
    for i in range(len(arr)):
        if arr[i] != 0:
            arr[non_zero_idx] = arr[i]
            non_zero_idx += 1

    # Fill remaining with zeroes
    while non_zero_idx < len(arr):
        arr[non_zero_idx] = 0
        non_zero_idx += 1

    return arr

def move_zeroes_to_start(arr):
    """Move all zeroes to start"""
    arr = arr.copy()
    non_zero_idx = len(arr) - 1

    for i in range(len(arr) - 1, -1, -1):
        if arr[i] != 0:
            arr[non_zero_idx] = arr[i]
            non_zero_idx -= 1

    while non_zero_idx >= 0:
        arr[non_zero_idx] = 0
        non_zero_idx -= 1

    return arr

# Examples
arr = [0, 1, 0, 3, 0, 5, 0, 7]

print("🔢 MOVE ZEROES")
print("=" * 45)
print(f"Original:       {arr}")
print(f"Zeroes to end:  {move_zeroes_to_end(arr)}")
print(f"Zeroes to start:{move_zeroes_to_start(arr)}")


# In[ ]:


def zeroes_calculator():
    """Interactive zeroes calculator"""

    print("🔢 ZEROES CALCULATOR")
    print("=" * 45)

    print("\n📋 OPTIONS:")
    print("1. Count zeroes in number")
    print("2. Count zeroes in array")
    print("3. Sum of zero indices")
    print("4. Trailing zeroes in factorial")
    print("5. Zeroes in binary")
    print("6. Move zeroes to end")
    print("7. Full analysis")

    choice = input("\nChoice (1-7): ")

    print("\n" + "=" * 45)

    if choice == '1':
        n = int(input("Enter number: "))
        count = str(abs(n)).count('0')
        print(f"Number: {n}")
        print(f"Zero count: {count}")

    elif choice == '2':
        arr_input = input("Enter array (space separated): ")
        arr = list(map(int, arr_input.split()))
        count = arr.count(0)
        print(f"Array: {arr}")
        print(f"Zero count: {count}")

    elif choice == '3':
        arr_input = input("Enter array (space separated): ")
        arr = list(map(int, arr_input.split()))
        indices = [i for i, x in enumerate(arr) if x == 0]
        print(f"Array: {arr}")
        print(f"Zero indices: {indices}")
        print(f"Sum of indices: {sum(indices)}")

    elif choice == '4':
        n = int(input("Enter n for n!: "))
        count = 0
        power = 5
        while power <= n:
            count += n // power
            power *= 5
        print(f"Trailing zeroes in {n}!: {count}")

    elif choice == '5':
        n = int(input("Enter number: "))
        binary = bin(n)[2:]
        zeroes = binary.count('0')
        print(f"{n} in binary: {binary}")
        print(f"Zero count: {zeroes}")

    elif choice == '6':
        arr_input = input("Enter array (space separated): ")
        arr = list(map(int, arr_input.split()))
        result = move_zeroes_to_end(arr)
        print(f"Original: {arr}")
        print(f"Zeroes at end: {result}")

    elif choice == '7':
        arr_input = input("Enter array (space separated): ")
        arr = list(map(int, arr_input.split()))
        zeroes_analysis_visualized(arr)

    print("=" * 45)

# Run
zeroes_calculator()


# In[ ]:


class ZeroesAnalyzer:
    """Complete utility for zeroes operations"""

    def __init__(self, data):
        """Initialize with number or array"""
        if isinstance(data, int):
            self.number = data
            self.array = None
        else:
            self.number = None
            self.array = list(data)

    # ===== NUMBER OPERATIONS =====

    def count_zeroes_in_number(self):
        """Count zeroes in number"""
        if self.number is None:
            return None
        return str(abs(self.number)).count('0')

    def trailing_zeroes_factorial(self):
        """Trailing zeroes in n!"""
        if self.number is None or self.number < 0:
            return None

        count = 0
        power = 5
        while power <= self.number:
            count += self.number // power
            power *= 5
        return count

    def zeroes_in_binary(self):
        """Zeroes in binary representation"""
        if self.number is None:
            return None
        return bin(abs(self.number))[2:].count('0')

    # ===== ARRAY OPERATIONS =====

    def count_zeroes_in_array(self):
        """Count zeroes in array"""
        if self.array is None:
            return None
        return self.array.count(0)

    def get_zero_indices(self):
        """Get indices of zeroes"""
        if self.array is None:
            return None
        return [i for i, x in enumerate(self.array) if x == 0]

    def sum_of_zero_indices(self):
        """Sum of indices where zeroes occur"""
        indices = self.get_zero_indices()
        return sum(indices) if indices else 0

    def move_zeroes_end(self):
        """Move zeroes to end"""
        if self.array is None:
            return None

        result = []
        zero_count = 0

        for num in self.array:
            if num == 0:
                zero_count += 1
            else:
                result.append(num)

        result.extend([0] * zero_count)
        return result

    def move_zeroes_start(self):
        """Move zeroes to start"""
        if self.array is None:
            return None

        zero_count = self.array.count(0)
        non_zeroes = [x for x in self.array if x != 0]
        return [0] * zero_count + non_zeroes

    def zero_percentage(self):
        """Percentage of zeroes"""
        if self.array is None or len(self.array) == 0:
            return None
        return (self.count_zeroes_in_array() / len(self.array)) * 100

    def full_analysis(self):
        """Complete analysis"""
        print("\n" + "╔" + "═" * 55 + "╗")
        print("║" + "🔢 ZEROES ANALYSIS".center(55) + "║")
        print("╠" + "═" * 55 + "╣")

        if self.number is not None:
            print(f"║  📝 Number: {self.number}".ljust(56) + "║")
            print("╠" + "═" * 55 + "╣")
            print(f"║  🔢 Zeroes in number: {self.count_zeroes_in_number()}".ljust(56) + "║")
            print(f"║  📊 Binary: {bin(abs(self.number))[2:]}".ljust(56) + "║")
            print(f"║  🔢 Zeroes in binary: {self.zeroes_in_binary()}".ljust(56) + "║")
            print(f"║  📊 Trailing zeroes in {self.number}!: {self.trailing_zeroes_factorial()}".ljust(56) + "║")

        if self.array is not None:
            arr_str = str(self.array)
            if len(arr_str) > 40:
                arr_str = arr_str[:37] + "..."
            print(f"║  📝 Array: {arr_str}".ljust(56) + "║")
            print(f"║  📏 Length: {len(self.array)}".ljust(56) + "║")
            print("╠" + "═" * 55 + "╣")
            print(f"║  🔢 Zero count: {self.count_zeroes_in_array()}".ljust(56) + "║")
            print(f"║  📍 Zero indices: {self.get_zero_indices()}".ljust(56) + "║")
            print(f"║  ➕ Sum of indices: {self.sum_of_zero_indices()}".ljust(56) + "║")
            print(f"║  📊 Zero percentage: {self.zero_percentage():.1f}%".ljust(56) + "║")
            print("╠" + "═" * 55 + "╣")

            moved = self.move_zeroes_end()
            moved_str = str(moved)
            if len(moved_str) > 35:
                moved_str = moved_str[:32] + "..."
            print(f"║  🔄 Zeroes at end: {moved_str}".ljust(56) + "║")

        print("╚" + "═" * 55 + "╝")


# Usage Examples
print("📊 NUMBER ANALYSIS:")
analyzer1 = ZeroesAnalyzer(10203)
analyzer1.full_analysis()

print("\n📊 ARRAY ANALYSIS:")
analyzer2 = ZeroesAnalyzer([1, 0, 2, 0, 3, 0, 4])
analyzer2.full_analysis()


# In[ ]:


arr = [1, 0, 2, 0, 3, 0]
n = 10203

# Count zeroes in array
count = arr.count(0)
print(f"Zeroes in array: {count}")

# Count zeroes in number
count = str(n).count('0')
print(f"Zeroes in number: {count}")

# Sum of zero indices
index_sum = sum(i for i, x in enumerate(arr) if x == 0)
print(f"Sum of zero indices: {index_sum}")

# Get zero indices
indices = [i for i, x in enumerate(arr) if x == 0]
print(f"Zero indices: {indices}")

# Move zeroes to end
moved = [x for x in arr if x != 0] + [0] * arr.count(0)
print(f"Zeroes at end: {moved}")

# Trailing zeroes in factorial
trailing = sum(n // 5**i for i in range(1, 20) if 5**i <= n)
print(f"Trailing zeroes in {n}!: {trailing}")

# Zeroes in binary
binary_zeroes = bin(n).count('0') - 1  # -1 for '0b' prefix
print(f"Zeroes in binary of {n}: {binary_zeroes}")

# Zero percentage
percentage = arr.count(0) / len(arr) * 100
print(f"Zero percentage: {percentage:.1f}%")

