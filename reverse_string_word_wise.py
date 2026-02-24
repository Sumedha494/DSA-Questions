#!/usr/bin/env python
# coding: utf-8

# In[1]:


def reverse_words_simple(s):
    """
    Reverse words using split and join
    Time: O(n), Space: O(n)
    """
    words = s.split()           # Split into words
    reversed_words = words[::-1] # Reverse the list
    return " ".join(reversed_words)  # Join back

# Example
s = "Hello World Python"
result = reverse_words_simple(s)

print(f"Original: '{s}'")
print(f"Reversed: '{result}'")


# In[2]:


s = "Hello World Python"

# Method 1: Split, reverse, join
result = " ".join(s.split()[::-1])
print(f"Method 1: {result}")

# Method 2: Using reversed()
result = " ".join(reversed(s.split()))
print(f"Method 2: {result}")

# Method 3: Using list reverse
words = s.split()
words.reverse()
result = " ".join(words)
print(f"Method 3: {result}")


# In[3]:


def reverse_words_visualized(s):
    """
    Visualize the word reversal process
    """
    print("🔄 REVERSE STRING WORD BY WORD")
    print("=" * 50)
    print(f"Original String: '{s}'")
    print()

    # Step 1: Split into words
    words = s.split()
    print(f"Step 1 - Split into words:")
    for i, word in enumerate(words):
        print(f"   Word {i}: '{word}'")
    print(f"   Words List: {words}")
    print()

    # Step 2: Reverse the list
    reversed_words = words[::-1]
    print(f"Step 2 - Reverse the list:")
    print(f"   Before: {words}")
    print(f"   After:  {reversed_words}")
    print()

    # Step 3: Join back
    result = " ".join(reversed_words)
    print(f"Step 3 - Join with space:")
    print(f"   Result: '{result}'")
    print()

    print("=" * 50)
    print(f"✅ Final Output: '{result}'")

    return result

# Example
s = "I love Python programming"
reverse_words_visualized(s)


# In[4]:


def reverse_words_loop(s):
    """
    Reverse words using for loop
    Time: O(n), Space: O(n)
    """
    words = s.split()
    result = []

    # Traverse from end to beginning
    for i in range(len(words) - 1, -1, -1):
        result.append(words[i])

    return " ".join(result)

# Example
s = "Hello World Python"

print(f"Original: '{s}'")
print(f"Reversed: '{reverse_words_loop(s)}'")


# In[5]:


def reverse_words_while(s):
    """
    Reverse words using while loop
    """
    words = s.split()
    left = 0
    right = len(words) - 1

    # Swap words from both ends
    while left < right:
        words[left], words[right] = words[right], words[left]
        left += 1
        right -= 1

    return " ".join(words)

# Example
s = "The quick brown fox"

print(f"Original: '{s}'")
print(f"Reversed: '{reverse_words_while(s)}'")


# In[6]:


def reverse_words_recursive(s):
    """
    Reverse words using recursion
    """
    words = s.split()

    def reverse(word_list):
        # Base case
        if len(word_list) <= 1:
            return word_list

        # Recursive case: last word + reverse of rest
        return [word_list[-1]] + reverse(word_list[:-1])

    reversed_words = reverse(words)
    return " ".join(reversed_words)

# Example
s = "Python is awesome"

print(f"Original: '{s}'")
print(f"Reversed: '{reverse_words_recursive(s)}'")


# In[7]:


def reverse_words_stack(s):
    """
    Reverse words using stack (Last In, First Out)
    """
    words = s.split()
    stack = []

    # Push all words to stack
    for word in words:
        stack.append(word)

    result = []

    # Pop all words from stack
    while stack:
        result.append(stack.pop())

    return " ".join(result)

# Example with visualization
s = "One Two Three Four"

print(f"Original: '{s}'")
print()

words = s.split()
stack = []

print("📥 Push to Stack:")
for word in words:
    stack.append(word)
    print(f"   Push '{word}' → Stack: {stack}")

print()
print("📤 Pop from Stack:")
result = []
while stack:
    word = stack.pop()
    result.append(word)
    print(f"   Pop '{word}' → Result: {result}")

print()
print(f"✅ Reversed: '{' '.join(result)}'")


# In[8]:


from collections import deque

def reverse_words_deque(s):
    """
    Reverse words using deque
    """
    words = s.split()
    d = deque(words)

    result = []

    # Pop from right (end)
    while d:
        result.append(d.pop())

    return " ".join(result)

# Alternative: appendleft
def reverse_words_deque_v2(s):
    words = s.split()
    d = deque()

    for word in words:
        d.appendleft(word)  # Add to front

    return " ".join(d)

# Example
s = "Hello World"

print(f"Original: '{s}'")
print(f"Method 1: '{reverse_words_deque(s)}'")
print(f"Method 2: '{reverse_words_deque_v2(s)}'")


# In[9]:


def reverse_words_clean(s):
    """
    Handle multiple spaces, leading/trailing spaces
    """
    # Method 1: split() automatically handles multiple spaces
    words = s.split()  # split() without argument splits on any whitespace
    return " ".join(reversed(words))

def reverse_words_preserve_spaces(s):
    """
    Reverse words but try to preserve spacing pattern
    """
    import re

    # Find all words and spaces
    tokens = re.split(r'(\s+)', s)

    # Separate words and spaces
    words = [t for t in tokens if t.strip()]
    spaces = [t for t in tokens if not t.strip()]

    # Reverse words
    words.reverse()

    # Reconstruct (simplified - just single spaces)
    return " ".join(words)

# Examples
test_strings = [
    "Hello World",
    "  Hello   World  ",
    "   Multiple   spaces   here   ",
    "Single",
    "   ",
]

print("🔄 HANDLE MULTIPLE SPACES")
print("=" * 55)

for s in test_strings:
    result = reverse_words_clean(s)
    print(f"Input:  '{s}'")
    print(f"Output: '{result}'")
    print("-" * 40)


# In[10]:


def reverse_words_inplace(s):
    """
    In-place reversal using character list

    Algorithm:
    1. Reverse entire string
    2. Reverse each word individually

    Time: O(n), Space: O(n) for the list
    """
    # Convert to list (strings are immutable in Python)
    chars = list(s)
    n = len(chars)

    def reverse_range(arr, start, end):
        """Reverse characters from start to end"""
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1

    # Step 1: Reverse entire string
    reverse_range(chars, 0, n - 1)
    print(f"After full reverse: '{''.join(chars)}'")

    # Step 2: Reverse each word
    start = 0
    for i in range(n + 1):
        if i == n or chars[i] == ' ':
            reverse_range(chars, start, i - 1)
            print(f"After reversing word: '{''.join(chars)}'")
            start = i + 1

    return ''.join(chars)

# Example
s = "Hello World"
print(f"Original: '{s}'")
print()
result = reverse_words_inplace(s)
print()
print(f"✅ Final: '{result}'")


# In[11]:


def reverse_words_and_letters(s):
    """
    Reverse word order AND reverse letters in each word
    """
    words = s.split()

    # Reverse each word's letters
    reversed_letters = [word[::-1] for word in words]

    # Reverse word order
    reversed_words = reversed_letters[::-1]

    return " ".join(reversed_words)

def reverse_only_words_order(s):
    """Reverse only word order"""
    return " ".join(s.split()[::-1])

def reverse_only_letters(s):
    """Reverse only letters in each word, keep word order"""
    return " ".join(word[::-1] for word in s.split())

# Example
s = "Hello World"

print(f"Original: '{s}'")
print(f"Reverse word order only:   '{reverse_only_words_order(s)}'")
print(f"Reverse letters only:      '{reverse_only_letters(s)}'")
print(f"Reverse both:              '{reverse_words_and_letters(s)}'")


# In[12]:


def reverse_words_safe(s):
    """
    Handle all edge cases
    """
    # Edge case: None or not a string
    if s is None or not isinstance(s, str):
        return ""

    # Edge case: Empty string
    if not s.strip():
        return ""

    # Split, reverse, join
    words = s.split()

    # Edge case: Single word
    if len(words) == 1:
        return words[0]

    return " ".join(reversed(words))

# Test cases
test_cases = [
    "Hello World",           # Normal
    "  Hello   World  ",     # Multiple spaces
    "Single",                # Single word
    "",                      # Empty
    "   ",                   # Only spaces
    None,                    # None
    "A B C D E",             # Multiple words
    "123 456 789",           # Numbers
    "Hello, World!",         # With punctuation
]

print("🔍 EDGE CASES HANDLING")
print("=" * 55)

for s in test_cases:
    result = reverse_words_safe(s)
    s_display = f"'{s}'" if s is not None else "None"
    print(f"Input: {s_display:<25} → Output: '{result}'")


# In[14]:


def word_reverser():
    """Interactive word reverser"""

    print("🔄 WORD REVERSER")
    print("=" * 45)

    s = input("Enter a string: ")

    print("\n📋 OPTIONS:")
    print("1. Reverse word order")
    print("2. Reverse letters in each word")
    print("3. Reverse both (words + letters)")
    print("4. Show all variations")

    choice = input("\nChoice (1-4): ")

    print("\n" + "=" * 45)
    print(f"Original: '{s}'")
    print("-" * 45)

    if choice == '1':
        result = " ".join(s.split()[::-1])
        print(f"Word order reversed: '{result}'")

    elif choice == '2':
        result = " ".join(word[::-1] for word in s.split())
        print(f"Letters reversed: '{result}'")

    elif choice == '3':
        words = [word[::-1] for word in s.split()]
        result = " ".join(words[::-1])
        print(f"Both reversed: '{result}'")

    elif choice == '4':
        print(f"Word order reversed: '{' '.join(s.split()[::-1])}'")
        print(f"Letters reversed:    '{' '.join(word[::-1] for word in s.split())}'")
        words = [word[::-1] for word in s.split()]
        print(f"Both reversed:       '{' '.join(words[::-1])}'")

    print("=" * 45)

# Run
word_reverser()


# In[15]:


class WordReverser:
    """Complete utility for word reversal operations"""

    def __init__(self, s):
        self.original = s
        self.words = s.split() if s else []

    def reverse_word_order(self):
        """Reverse the order of words"""
        return " ".join(self.words[::-1])

    def reverse_each_word(self):
        """Reverse letters in each word"""
        return " ".join(word[::-1] for word in self.words)

    def reverse_both(self):
        """Reverse word order and letters"""
        reversed_letters = [word[::-1] for word in self.words]
        return " ".join(reversed_letters[::-1])

    def reverse_first_last(self):
        """Swap only first and last words"""
        if len(self.words) < 2:
            return self.original

        words = self.words.copy()
        words[0], words[-1] = words[-1], words[0]
        return " ".join(words)

    def reverse_alternate(self):
        """Reverse alternate words"""
        result = []
        for i, word in enumerate(self.words):
            if i % 2 == 1:
                result.append(word[::-1])
            else:
                result.append(word)
        return " ".join(result)

    def capitalize_reversed(self):
        """Reverse and capitalize each word"""
        reversed_words = self.words[::-1]
        return " ".join(word.capitalize() for word in reversed_words)

    def get_statistics(self):
        """Get string statistics"""
        return {
            'total_chars': len(self.original),
            'word_count': len(self.words),
            'words': self.words,
            'word_lengths': [len(w) for w in self.words],
            'longest_word': max(self.words, key=len) if self.words else "",
            'shortest_word': min(self.words, key=len) if self.words else "",
        }

    def full_analysis(self):
        """Display all reversals and statistics"""
        stats = self.get_statistics()

        print("\n" + "╔" + "═" * 55 + "╗")
        print("║" + "🔄 WORD REVERSER - COMPLETE ANALYSIS".center(55) + "║")
        print("╠" + "═" * 55 + "╣")

        print(f"║  📝 Original: '{self.original}'".ljust(56) + "║")
        print(f"║  📏 Length: {stats['total_chars']} characters".ljust(56) + "║")
        print(f"║  📊 Words: {stats['word_count']}".ljust(56) + "║")
        print("╠" + "═" * 55 + "╣")

        print("║  📋 WORD LIST:".ljust(56) + "║")
        for i, word in enumerate(stats['words']):
            print(f"║     {i+1}. '{word}' ({len(word)} chars)".ljust(56) + "║")

        print("╠" + "═" * 55 + "╣")
        print("║  🔄 REVERSALS:".ljust(56) + "║")
        print(f"║     Word Order:    '{self.reverse_word_order()}'".ljust(56) + "║")
        print(f"║     Each Word:     '{self.reverse_each_word()}'".ljust(56) + "║")
        print(f"║     Both:          '{self.reverse_both()}'".ljust(56) + "║")
        print(f"║     First-Last:    '{self.reverse_first_last()}'".ljust(56) + "║")
        print(f"║     Alternate:     '{self.reverse_alternate()}'".ljust(56) + "║")
        print(f"║     Capitalized:   '{self.capitalize_reversed()}'".ljust(56) + "║")

        print("╚" + "═" * 55 + "╝")


# Usage
s = "Hello Beautiful World"
reverser = WordReverser(s)
reverser.full_analysis()


# In[16]:


import re

def reverse_words_regex(s):
    """
    Reverse words using regex for word extraction
    """
    # Find all words (alphanumeric sequences)
    words = re.findall(r'\S+', s)

    return " ".join(reversed(words))

def reverse_keeping_punctuation(s):
    """
    Reverse words but handle punctuation smartly
    """
    # Match words with attached punctuation
    pattern = r'(\w+)([^\w\s]*)'

    matches = re.findall(pattern, s)

    # Reverse the words, keep punctuation attached
    reversed_matches = matches[::-1]

    result = []
    for word, punct in reversed_matches:
        result.append(word + punct)

    return " ".join(result)

# Examples
test_strings = [
    "Hello, World!",
    "How are you?",
    "Python is great!",
]

print("🔤 REVERSE WITH PUNCTUATION")
print("=" * 50)

for s in test_strings:
    result1 = reverse_words_regex(s)
    result2 = reverse_keeping_punctuation(s)

    print(f"Original: '{s}'")
    print(f"Simple:   '{result1}'")
    print(f"Smart:    '{result2}'")
    print("-" * 40)


# In[17]:


def reverse_k_words(s, k):
    """
    Reverse every k words
    """
    words = s.split()
    result = []

    for i in range(0, len(words), k):
        chunk = words[i:i+k]
        result.extend(chunk[::-1])

    return " ".join(result)

# Example
s = "A B C D E F G H"

print(f"Original: '{s}'")
print()

for k in range(1, 5):
    result = reverse_k_words(s, k)
    print(f"Reverse every {k} words: '{result}'")


# In[18]:


import time
from collections import deque

def measure_time(func, s, iterations=10000):
    """Measure execution time"""
    start = time.time()
    for _ in range(iterations):
        func(s)
    end = time.time()
    return (end - start) / iterations * 1000

# Different methods
def method_split_join(s):
    return " ".join(s.split()[::-1])

def method_reversed(s):
    return " ".join(reversed(s.split()))

def method_loop(s):
    words = s.split()
    result = []
    for i in range(len(words) - 1, -1, -1):
        result.append(words[i])
    return " ".join(result)

def method_deque(s):
    words = s.split()
    d = deque()
    for word in words:
        d.appendleft(word)
    return " ".join(d)

# Test
s = "The quick brown fox jumps over the lazy dog"

print("⚡ PERFORMANCE COMPARISON")
print("=" * 50)
print(f"String: '{s}'")
print("-" * 50)

methods = [
    ("split()[::-1]", method_split_join),
    ("reversed()", method_reversed),
    ("For Loop", method_loop),
    ("Deque", method_deque),
]

for name, func in methods:
    time_ms = measure_time(func, s)
    result = func(s)[:20] + "..."
    print(f"{name:<15}: {time_ms:.4f} ms → '{result}'")

print("-" * 50)
print("✅ split()[::-1] and reversed() are fastest!")


# In[19]:


s = "Hello World Python"

# Method 1: Slicing (Most common)
print(" ".join(s.split()[::-1]))

# Method 2: Using reversed()
print(" ".join(reversed(s.split())))

# Method 3: Using reduce
from functools import reduce
print(reduce(lambda x, y: y + " " + x, s.split()))

# Method 4: Using recursion lambda
rev = lambda s: s if ' ' not in s else rev(s[s.find(' ')+1:]) + ' ' + s[:s.find(' ')]
print(rev(s))

# Method 5: Using list comprehension
words = s.split()
print(" ".join([words[i] for i in range(len(words)-1, -1, -1)]))


# In[ ]:




