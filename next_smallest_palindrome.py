#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def is_palindrome(n):
    """Check if number is palindrome"""
    s = str(n)
    return s == s[::-1]

def next_palindrome_brute(n):
    """
    Find next palindrome by incrementing
    Time: O(n * √n) - Slow for large numbers
    """
    n = n + 1

    while not is_palindrome(n):
        n += 1

    return n

# Examples
test_numbers = [123, 999, 1221, 12321, 99, 10, 1]

print("🔢 NEXT PALINDROME (Brute Force)")
print("=" * 40)

for n in test_numbers:
    result = next_palindrome_brute(n)
    print(f"{n} → {result}")


# In[ ]:


def next_palindrome_efficient(n):
    """
    Efficient method using mirroring
    Time: O(d) where d = number of digits
    """
    s = str(n)
    length = len(s)

    # Edge case: All 9s (999 → 1001)
    if all(c == '9' for c in s):
        return int('1' + '0' * (length - 1) + '1')

    # Convert to list for manipulation
    digits = list(s)

    # Find middle
    mid = length // 2

    # Mirror left half to right half
    left = mid - 1
    right = mid if length % 2 == 0 else mid + 1

    # Copy left to right (mirror)
    while left >= 0:
        digits[right] = digits[left]
        left -= 1
        right += 1

    # Check if mirrored number is greater than original
    mirrored = int(''.join(digits))

    if mirrored > n:
        return mirrored

    # If not greater, increment middle part
    digits = list(str(n))

    # Start from middle and carry over
    carry = 1
    left = mid - 1
    right = mid if length % 2 == 0 else mid + 1
    mid_idx = mid if length % 2 == 1 else -1

    # Handle odd length - increment middle first
    if length % 2 == 1:
        val = int(digits[mid]) + carry
        carry = val // 10
        digits[mid] = str(val % 10)

    # Propagate carry and mirror
    while left >= 0:
        val = int(digits[left]) + carry
        carry = val // 10
        digits[left] = str(val % 10)
        digits[right] = digits[left]
        left -= 1
        right += 1

    # If still carry, we need extra digit
    if carry:
        return int('1' + '0' * (length - 1) + '1')

    return int(''.join(digits))

# Examples
test_numbers = [123, 999, 1221, 12321, 99, 808, 12345]

print("🔢 NEXT PALINDROME (Efficient)")
print("=" * 40)

for n in test_numbers:
    result = next_palindrome_efficient(n)
    print(f"{n} → {result}")


# In[ ]:


def next_palindrome_visualized(n):
    """
    Visualize the palindrome finding process
    """
    print("🔢 FINDING NEXT PALINDROME")
    print("=" * 55)
    print(f"Input Number: {n}")

    s = str(n)
    length = len(s)

    print(f"Digits: {list(s)}")
    print(f"Length: {length}")
    print("-" * 55)

    # Check for all 9s
    if all(c == '9' for c in s):
        result = int('1' + '0' * (length - 1) + '1')
        print(f"⚠️ All 9s detected!")
        print(f"Result: {result}")
        return result

    # Step 1: Mirror left to right
    print("\n📊 STEP 1: Mirror left half to right")

    digits = list(s)
    mid = length // 2

    print(f"Middle index: {mid}")
    print(f"Original: {digits}")

    left = mid - 1
    right = mid if length % 2 == 0 else mid + 1

    while left >= 0:
        print(f"   Copy digits[{left}]='{digits[left]}' to digits[{right}]")
        digits[right] = digits[left]
        left -= 1
        right += 1

    mirrored = int(''.join(digits))
    print(f"Mirrored: {digits} = {mirrored}")

    # Check if greater
    if mirrored > n:
        print(f"\n✅ Mirrored ({mirrored}) > Original ({n})")
        print(f"Result: {mirrored}")
        return mirrored

    # Step 2: Need to increment middle
    print(f"\n❌ Mirrored ({mirrored}) <= Original ({n})")
    print("\n📊 STEP 2: Increment middle and mirror again")

    digits = list(s)
    carry = 1
    left = mid - 1
    right = mid if length % 2 == 0 else mid + 1

    # Handle odd length middle
    if length % 2 == 1:
        val = int(digits[mid]) + carry
        carry = val // 10
        digits[mid] = str(val % 10)
        print(f"   Increment middle: digits[{mid}] = {digits[mid]}")

    # Propagate carry
    while left >= 0 and carry:
        val = int(digits[left]) + carry
        carry = val // 10
        digits[left] = str(val % 10)
        digits[right] = digits[left]
        print(f"   Update digits[{left}] and digits[{right}] = {digits[left]}")
        left -= 1
        right += 1

    # Mirror remaining
    while left >= 0:
        digits[right] = digits[left]
        left -= 1
        right += 1

    if carry:
        result = int('1' + '0' * (length - 1) + '1')
    else:
        result = int(''.join(digits))

    print(f"\nFinal digits: {digits}")
    print(f"✅ Result: {result}")

    return result

# Examples
test_numbers = [1234, 1991, 999, 12321]

for n in test_numbers:
    next_palindrome_visualized(n)
    print("\n" + "=" * 55 + "\n")


# In[ ]:


def next_palindrome_complete(n):
    """
    Complete solution handling all edge cases
    """
    # Edge case: Negative numbers
    if n < 0:
        return 0  # First positive palindrome

    # Edge case: Single digit
    if n < 9:
        return n + 1

    if n == 9:
        return 11

    s = str(n)
    length = len(s)

    # Edge case: All 9s
    if all(c == '9' for c in s):
        return int('1' + '0' * (length - 1) + '1')

    digits = list(s)
    mid = length // 2
    is_odd = length % 2 == 1

    # Mirror left to right
    for i in range(mid):
        digits[length - 1 - i] = digits[i]

    mirrored = int(''.join(digits))

    if mirrored > n:
        return mirrored

    # Increment from middle
    digits = list(s)
    carry = 1

    # Start position for increment
    i = mid if is_odd else mid - 1

    while i >= 0 and carry:
        val = int(digits[i]) + carry
        carry = val // 10
        digits[i] = str(val % 10)
        i -= 1

    # Mirror again
    for i in range(mid):
        digits[length - 1 - i] = digits[i]

    if carry:
        return int('1' + '0' * (length - 1) + '1')

    return int(''.join(digits))

# Test all edge cases
test_cases = [
    0,        # Zero
    5,        # Single digit
    9,        # Nine
    10,       # Two digits
    99,       # Two 9s
    100,      # Three digits starting with 1
    999,      # Three 9s
    1001,     # Already palindrome
    1234,     # Normal case
    12321,    # Already palindrome (odd)
    12345,    # Normal (odd length)
    99999,    # All 9s
    123456789,# Large number
]

print("🔢 NEXT PALINDROME - ALL EDGE CASES")
print("=" * 50)

for n in test_cases:
    result = next_palindrome_complete(n)
    is_pal = str(result) == str(result)[::-1]
    status = "✅" if is_pal else "❌"
    print(f"{n:>12} → {result:>12} {status}")


# In[ ]:


def next_palindrome_array(digits):
    """
    Find next palindrome for array of digits
    Input: [1, 2, 3]
    Output: [1, 3, 1]
    """
    n = len(digits)

    if n == 0:
        return [1]

    # Check all 9s
    if all(d == 9 for d in digits):
        return [1] + [0] * (n - 1) + [1]

    result = digits.copy()
    mid = n // 2
    is_odd = n % 2 == 1

    # Mirror left to right
    for i in range(mid):
        result[n - 1 - i] = result[i]

    # Check if greater
    if result > digits:
        return result

    # Increment middle part
    result = digits.copy()
    carry = 1

    i = mid if is_odd else mid - 1

    while i >= 0 and carry:
        val = result[i] + carry
        carry = val // 10
        result[i] = val % 10
        i -= 1

    # Mirror again
    for i in range(mid):
        result[n - 1 - i] = result[i]

    if carry:
        return [1] + [0] * (n - 1) + [1]

    return result

# Examples
test_arrays = [
    [1, 2, 3],
    [9, 9, 9],
    [1, 2, 2, 1],
    [1, 2, 3, 4, 5],
    [1],
    [9],
]

print("🔢 NEXT PALINDROME (Array Input)")
print("=" * 45)

for arr in test_arrays:
    result = next_palindrome_array(arr)
    print(f"{arr} → {result}")


# In[ ]:


def next_k_palindromes(n, k):
    """
    Find k palindromes greater than n
    """
    palindromes = []
    current = n

    for _ in range(k):
        current = next_palindrome_complete(current)
        palindromes.append(current)

    return palindromes

# Example
n = 100
k = 10

palindromes = next_k_palindromes(n, k)

print(f"🔢 NEXT {k} PALINDROMES AFTER {n}")
print("=" * 40)

for i, p in enumerate(palindromes, 1):
    print(f"{i:2}. {p}")


# In[ ]:


def previous_palindrome(n):
    """
    Find the largest palindrome smaller than n
    """
    if n <= 0:
        return None

    if n <= 10:
        return n - 1 if n > 1 else None

    s = str(n - 1)
    length = len(s)

    # Start with n-1 and find palindrome
    digits = list(s)
    mid = length // 2
    is_odd = length % 2 == 1

    # Mirror left to right
    for i in range(mid):
        digits[length - 1 - i] = digits[i]

    mirrored = int(''.join(digits))

    if mirrored < n:
        return mirrored

    # Decrement middle
    digits = list(s)
    borrow = 1

    i = mid if is_odd else mid - 1

    while i >= 0 and borrow:
        val = int(digits[i]) - borrow
        if val < 0:
            val = 9
            borrow = 1
        else:
            borrow = 0
        digits[i] = str(val)
        i -= 1

    # Remove leading zeros
    result_str = ''.join(digits).lstrip('0')

    if not result_str:
        return 0

    # Mirror again
    digits = list(result_str)
    new_len = len(digits)
    new_mid = new_len // 2

    for i in range(new_mid):
        digits[new_len - 1 - i] = digits[i]

    return int(''.join(digits))

# Examples
test_numbers = [123, 1000, 12321, 100, 11, 10]

print("🔢 PREVIOUS PALINDROME")
print("=" * 40)

for n in test_numbers:
    prev = previous_palindrome(n)
    next_pal = next_palindrome_complete(n)
    print(f"Prev ← {prev:>6} | {n:>6} | {next_pal:>6} → Next")


# In[ ]:


def closest_palindrome(n):
    """
    Find the closest palindrome to n (not equal to n)
    LeetCode 564
    """
    if n <= 10:
        return n - 1 if n > 0 else 1

    s = str(n)
    length = len(s)

    candidates = set()

    # Candidate 1: All 9s with one less digit (99, 999, ...)
    candidates.add(int('9' * (length - 1)))

    # Candidate 2: 10...01 with one more digit (1001, 10001, ...)
    candidates.add(int('1' + '0' * (length - 1) + '1'))

    # Candidate 3: Mirror with same, +1, -1 middle
    prefix = int(s[:(length + 1) // 2])

    for diff in [-1, 0, 1]:
        new_prefix = str(prefix + diff)

        if length % 2 == 0:
            candidate = new_prefix + new_prefix[::-1]
        else:
            candidate = new_prefix + new_prefix[-2::-1]

        candidates.add(int(candidate))

    # Remove n itself
    candidates.discard(n)

    # Find closest
    closest = min(candidates, key=lambda x: (abs(x - n), x))

    return closest

# Examples
test_numbers = [123, 1000, 12321, 99, 100, 1234]

print("🔢 CLOSEST PALINDROME")
print("=" * 50)

for n in test_numbers:
    closest = closest_palindrome(n)
    diff = abs(closest - n)
    print(f"{n} → {closest} (difference: {diff})")


# In[ ]:


def palindromes_in_range(start, end):
    """
    Generate all palindromes in range [start, end]
    """
    palindromes = []

    # Start from first palindrome >= start
    if start <= 0:
        current = 0
    else:
        current = start - 1
        current = next_palindrome_complete(current)

    while current <= end:
        palindromes.append(current)
        current = next_palindrome_complete(current)

    return palindromes

# Example
start, end = 100, 200

palindromes = palindromes_in_range(start, end)

print(f"🔢 PALINDROMES IN RANGE [{start}, {end}]")
print("=" * 40)
print(f"Count: {len(palindromes)}")
print(f"Palindromes: {palindromes}")


# In[ ]:


def palindrome_tool():
    """Interactive palindrome finder"""

    print("🔢 PALINDROME FINDER")
    print("=" * 45)

    n = int(input("Enter a number: "))

    print("\n📋 OPTIONS:")
    print("1. Next palindrome")
    print("2. Previous palindrome")
    print("3. Closest palindrome")
    print("4. Next K palindromes")
    print("5. Palindromes in range")
    print("6. Full analysis")

    choice = input("\nChoice (1-6): ")

    print("\n" + "=" * 45)

    if choice == '1':
        result = next_palindrome_complete(n)
        print(f"Next palindrome after {n}: {result}")

    elif choice == '2':
        result = previous_palindrome(n)
        print(f"Previous palindrome before {n}: {result}")

    elif choice == '3':
        result = closest_palindrome(n)
        print(f"Closest palindrome to {n}: {result}")

    elif choice == '4':
        k = int(input("How many? "))
        palindromes = next_k_palindromes(n, k)
        print(f"Next {k} palindromes after {n}:")
        for i, p in enumerate(palindromes, 1):
            print(f"  {i}. {p}")

    elif choice == '5':
        end = int(input("Enter end of range: "))
        palindromes = palindromes_in_range(n, end)
        print(f"Palindromes in [{n}, {end}]: {palindromes}")

    elif choice == '6':
        is_pal = str(n) == str(n)[::-1]
        next_pal = next_palindrome_complete(n)
        prev_pal = previous_palindrome(n)
        closest = closest_palindrome(n)

        print(f"\n📊 FULL ANALYSIS OF {n}:")
        print(f"   Is Palindrome: {'✅ Yes' if is_pal else '❌ No'}")
        print(f"   Next: {next_pal}")
        print(f"   Previous: {prev_pal}")
        print(f"   Closest: {closest}")
        print(f"   Next 5: {next_k_palindromes(n, 5)}")

    print("=" * 45)

# Run
palindrome_tool()


# In[ ]:


class PalindromeFinder:
    """Complete utility for palindrome operations"""

    def __init__(self, n):
        self.n = n
        self.s = str(n)
        self.length = len(self.s)

    def is_palindrome(self):
        """Check if number is palindrome"""
        return self.s == self.s[::-1]

    def next_palindrome(self):
        """Find next greater palindrome"""
        if self.n < 9:
            return self.n + 1

        if self.n == 9:
            return 11

        # Check all 9s
        if all(c == '9' for c in self.s):
            return int('1' + '0' * (self.length - 1) + '1')

        digits = list(self.s)
        mid = self.length // 2
        is_odd = self.length % 2 == 1

        # Mirror
        for i in range(mid):
            digits[self.length - 1 - i] = digits[i]

        mirrored = int(''.join(digits))

        if mirrored > self.n:
            return mirrored

        # Increment
        digits = list(self.s)
        carry = 1
        i = mid if is_odd else mid - 1

        while i >= 0 and carry:
            val = int(digits[i]) + carry
            carry = val // 10
            digits[i] = str(val % 10)
            i -= 1

        for i in range(mid):
            digits[self.length - 1 - i] = digits[i]

        if carry:
            return int('1' + '0' * (self.length - 1) + '1')

        return int(''.join(digits))

    def previous_palindrome(self):
        """Find previous smaller palindrome"""
        if self.n <= 1:
            return None

        for i in range(self.n - 1, 0, -1):
            if str(i) == str(i)[::-1]:
                return i

        return None

    def closest_palindrome(self):
        """Find closest palindrome (not equal)"""
        next_pal = self.next_palindrome()
        prev_pal = self.previous_palindrome() or 0

        if next_pal - self.n <= self.n - prev_pal:
            return next_pal
        return prev_pal

    def next_k(self, k):
        """Find next k palindromes"""
        result = []
        current = self.n

        for _ in range(k):
            finder = PalindromeFinder(current)
            current = finder.next_palindrome()
            result.append(current)

        return result

    def get_mirror(self):
        """Get the mirrored version"""
        digits = list(self.s)
        mid = self.length // 2

        for i in range(mid):
            digits[self.length - 1 - i] = digits[i]

        return int(''.join(digits))

    def distance_to_palindrome(self):
        """Distance to nearest palindrome"""
        if self.is_palindrome():
            return 0

        closest = self.closest_palindrome()
        return abs(self.n - closest)

    def full_analysis(self):
        """Complete analysis"""
        print("\n" + "╔" + "═" * 55 + "╗")
        print("║" + "🔢 PALINDROME ANALYSIS".center(55) + "║")
        print("╠" + "═" * 55 + "╣")

        print(f"║  📝 Number: {self.n}".ljust(56) + "║")
        print(f"║  📏 Digits: {self.length}".ljust(56) + "║")

        is_pal = self.is_palindrome()
        status = "✅ Yes" if is_pal else "❌ No"
        print(f"║  🔍 Is Palindrome: {status}".ljust(56) + "║")

        print("╠" + "═" * 55 + "╣")

        print(f"║  ⬆️  Next Palindrome: {self.next_palindrome()}".ljust(56) + "║")

        prev = self.previous_palindrome()
        print(f"║  ⬇️  Previous Palindrome: {prev if prev else 'None'}".ljust(56) + "║")
        print(f"║  🎯 Closest Palindrome: {self.closest_palindrome()}".ljust(56) + "║")
        print(f"║  📐 Distance to Nearest: {self.distance_to_palindrome()}".ljust(56) + "║")

        print("╠" + "═" * 55 + "╣")

        mirrored = self.get_mirror()
        print(f"║  🪞 Mirrored: {mirrored}".ljust(56) + "║")

        next_5 = self.next_k(5)
        print(f"║  📊 Next 5: {next_5}".ljust(56) + "║")

        print("╚" + "═" * 55 + "╝")


# Usage
test_numbers = [123, 999, 12321, 1234]

for n in test_numbers:
    finder = PalindromeFinder(n)
    finder.full_analysis()


# In[ ]:


# Check palindrome
is_pal = lambda n: str(n) == str(n)[::-1]

# Next palindrome (brute force)
next_pal = lambda n: next(i for i in range(n+1, n+10**len(str(n))) if str(i)==str(i)[::-1])

# Mirror number
mirror = lambda n: int(str(n)[:len(str(n))//2+len(str(n))%2] + str(n)[:len(str(n))//2][::-1])

# Count palindromes up to n
count_pal = lambda n: sum(1 for i in range(1, n+1) if str(i)==str(i)[::-1])

# Examples
n = 123
print(f"Is {n} palindrome: {is_pal(n)}")
print(f"Next palindrome: {next_pal(n)}")
print(f"Mirror of {n}: {mirror(n)}")
print(f"Palindromes up to {n}: {count_pal(n)}")

