#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def is_balanced(s):
    """
    Check if parentheses are balanced
    Time: O(n), Space: O(1)
    """
    count = 0

    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1

        # More closing than opening at any point
        if count < 0:
            return False

    # All opening should be closed
    return count == 0

# Examples
test_cases = [
    "()",
    "(())",
    "((()))",
    "((",
    "))",
    "())",
    "(()",
    "(()())",
    ")(", 
]

print("🔍 CHECK BALANCED PARENTHESES")
print("=" * 40)

for s in test_cases:
    result = is_balanced(s)
    status = "✅ Balanced" if result else "❌ Unbalanced"
    print(f"'{s}' → {status}")


# In[ ]:


def min_add_to_make_valid(s):
    """
    Find minimum parentheses to add to make string valid
    LeetCode 921

    Time: O(n), Space: O(1)
    """
    open_needed = 0   # Unmatched '(' - need ')'
    close_needed = 0  # Unmatched ')' - need '('

    for char in s:
        if char == '(':
            open_needed += 1
        elif char == ')':
            if open_needed > 0:
                open_needed -= 1  # Match with existing '('
            else:
                close_needed += 1  # Need '(' for this ')'

    return open_needed + close_needed

# Examples
test_cases = [
    "()",       # 0 - already balanced
    "(()",      # 1 - need 1 ')'
    "))((",     # 4 - need 2 '(' and 2 ')'
    "(((",      # 3 - need 3 ')'
    ")))",      # 3 - need 3 '('
    "(()())",   # 0 - already balanced
    "()))((",   # 4 - need to add
    "",         # 0 - empty is valid
]

print("📊 MINIMUM PARENTHESES TO ADD")
print("=" * 50)

for s in test_cases:
    result = min_add_to_make_valid(s)
    print(f"'{s}' → Add {result} parentheses")


# In[ ]:


def min_add_visualized(s):
    """
    Visualize the process of finding minimum additions
    """
    print("🔍 MINIMUM PARENTHESES TO ADD - VISUALIZATION")
    print("=" * 55)
    print(f"Input String: '{s}'")
    print("-" * 55)

    if not s:
        print("Empty string - already valid!")
        return 0

    open_needed = 0
    close_needed = 0

    print(f"{'Step':<6}{'Char':<6}{'Action':<30}{'Open':<8}{'Close'}")
    print("-" * 55)

    for i, char in enumerate(s):
        action = ""

        if char == '(':
            open_needed += 1
            action = "Found '(' - increment open"
        elif char == ')':
            if open_needed > 0:
                open_needed -= 1
                action = "Found ')' - matched with '('"
            else:
                close_needed += 1
                action = "Found ')' - no '(' to match"
        else:
            action = f"Skipping '{char}'"

        print(f"{i+1:<6}'{char}'   {action:<30}{open_needed:<8}{close_needed}")

    print("-" * 55)
    print(f"\n📊 RESULT:")
    print(f"   Unmatched '(' (need ')' for these): {open_needed}")
    print(f"   Unmatched ')' (need '(' for these): {close_needed}")
    print(f"   Total additions needed: {open_needed + close_needed}")

    # Show solution
    solution = '(' * close_needed + s + ')' * open_needed
    print(f"\n✅ Valid String: '{solution}'")

    return open_needed + close_needed

# Example
s = "()))(("
min_add_visualized(s)


# In[ ]:


def min_remove_to_make_valid(s):
    """
    Find minimum parentheses to remove to make string valid
    LeetCode 1249

    Returns: (count, resulting_string)
    Time: O(n), Space: O(n)
    """
    # First pass: mark invalid ')' 
    stack = []
    s_list = list(s)

    for i, char in enumerate(s_list):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()  # Match found
            else:
                s_list[i] = ''  # Mark for removal

    # Remaining in stack are unmatched '('
    for i in stack:
        s_list[i] = ''

    result = ''.join(s_list)
    removed = len(s) - len(result)

    return removed, result

# Examples
test_cases = [
    "()",
    "(()",
    "())",
    "(()())",
    "))((",
    "((())",
    "())()(",
    "a(b)c)d(",
]

print("📊 MINIMUM PARENTHESES TO REMOVE")
print("=" * 55)

for s in test_cases:
    count, result = min_remove_to_make_valid(s)
    print(f"'{s}' → Remove {count} → '{result}'")


# In[ ]:


def min_remove_visualized(s):
    """
    Visualize the removal process
    """
    print("🔍 MINIMUM PARENTHESES TO REMOVE - VISUALIZATION")
    print("=" * 60)
    print(f"Input String: '{s}'")
    print("-" * 60)

    stack = []
    s_list = list(s)
    invalid_indices = []

    print("\n📊 STEP 1: Find invalid ')' (no matching '(')")
    print("-" * 40)

    for i, char in enumerate(s_list):
        if char == '(':
            stack.append(i)
            print(f"   Index {i}: '(' - Push to stack. Stack: {stack}")
        elif char == ')':
            if stack:
                popped = stack.pop()
                print(f"   Index {i}: ')' - Match with index {popped}. Stack: {stack}")
            else:
                invalid_indices.append(i)
                print(f"   Index {i}: ')' - ❌ No match! Mark for removal")

    print("\n📊 STEP 2: Remaining '(' in stack are invalid")
    print("-" * 40)

    if stack:
        print(f"   Unmatched '(' at indices: {stack}")
        invalid_indices.extend(stack)
    else:
        print("   All '(' are matched!")

    print("\n📊 STEP 3: Remove invalid parentheses")
    print("-" * 40)
    print(f"   Invalid indices: {sorted(invalid_indices)}")

    # Build result
    result = ''.join(c for i, c in enumerate(s) if i not in invalid_indices)

    print("\n📊 RESULT:")
    print(f"   Original:  '{s}'")
    print(f"   Removed:   {len(invalid_indices)} parentheses")
    print(f"   Result:    '{result}'")

    # Visual marking
    marked = ""
    for i, c in enumerate(s):
        if i in invalid_indices:
            marked += f"[{c}]"
        else:
            marked += c
    print(f"   Marked:    '{marked}' ([] = removed)")

    return len(invalid_indices), result

# Example
s = "a(b(c)d)e)f("
min_remove_visualized(s)


# In[ ]:


def min_parentheses_stack(s):
    """
    Using stack to find minimum additions/removals
    """
    stack = []  # Store indices of unmatched '('
    unmatched_close = 0  # Count of unmatched ')'

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                unmatched_close += 1

    unmatched_open = len(stack)

    return {
        'unmatched_open': unmatched_open,
        'unmatched_close': unmatched_close,
        'total_to_add': unmatched_open + unmatched_close,
        'to_add_open': unmatched_close,  # Need '(' for unmatched ')'
        'to_add_close': unmatched_open,  # Need ')' for unmatched '('
    }

# Example
test_cases = [
    "(())",
    "(()",
    "())",
    "))((",
    "((())",
    "(()()(",
]

print("📊 DETAILED PARENTHESES ANALYSIS")
print("=" * 60)

for s in test_cases:
    result = min_parentheses_stack(s)

    print(f"\nString: '{s}'")
    print(f"   Unmatched '(': {result['unmatched_open']}")
    print(f"   Unmatched ')': {result['unmatched_close']}")
    print(f"   Need to add '(': {result['to_add_open']}")
    print(f"   Need to add ')': {result['to_add_close']}")
    print(f"   Total additions: {result['total_to_add']}")


# In[ ]:


def min_brackets_to_add(s):
    """
    Handle multiple bracket types: (), [], {}
    """
    matching = {')': '(', ']': '[', '}': '{'}
    opening = set('([{')
    closing = set(')]}')

    stack = []
    additions = 0

    for char in s:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == matching[char]:
                stack.pop()
            else:
                additions += 1  # Need matching opening bracket

    # Remaining opening brackets need closing
    additions += len(stack)

    return additions

def validate_brackets(s):
    """
    Validate multiple bracket types
    """
    matching = {')': '(', ']': '[', '}': '{'}
    stack = []

    for i, char in enumerate(s):
        if char in '([{':
            stack.append((char, i))
        elif char in ')]}':
            if stack and stack[-1][0] == matching[char]:
                stack.pop()
            else:
                return False, f"Unmatched '{char}' at position {i}"

    if stack:
        return False, f"Unmatched '{stack[-1][0]}' at position {stack[-1][1]}"

    return True, "All brackets matched"

# Examples
test_cases = [
    "()",
    "()[]{}",
    "([{}])",
    "([)]",
    "{[}]",
    "(([])){",
    "",
]

print("📊 MULTIPLE BRACKET TYPES")
print("=" * 55)

for s in test_cases:
    additions = min_brackets_to_add(s)
    valid, msg = validate_brackets(s)
    status = "✅" if valid else "❌"

    print(f"'{s}'")
    print(f"   Valid: {status} {msg}")
    print(f"   Min to add: {additions}")
    print()


# In[ ]:


def make_valid(s):
    """
    Return the valid string by adding minimum parentheses
    """
    # Count what we need
    open_needed = 0
    close_needed = 0

    for char in s:
        if char == '(':
            open_needed += 1
        elif char == ')':
            if open_needed > 0:
                open_needed -= 1
            else:
                close_needed += 1

    # Add required parentheses
    prefix = '(' * close_needed  # Add '(' at beginning
    suffix = ')' * open_needed   # Add ')' at end

    return prefix + s + suffix

def make_valid_by_removal(s):
    """
    Return the valid string by removing minimum parentheses
    """
    # First pass: remove invalid ')'
    stack = []
    chars = list(s)

    for i, char in enumerate(chars):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                chars[i] = ''

    # Second pass: remove unmatched '('
    for i in stack:
        chars[i] = ''

    return ''.join(chars)

# Examples
test_cases = [
    "((",
    "))",
    "(()",
    "())",
    "))((",
    "a(b)c)d(",
]

print("📊 MAKE VALID PARENTHESES")
print("=" * 60)
print(f"{'Original':<15}{'Add Method':<20}{'Remove Method':<20}")
print("-" * 60)

for s in test_cases:
    by_add = make_valid(s)
    by_remove = make_valid_by_removal(s)
    print(f"'{s}'".ljust(15) + f"'{by_add}'".ljust(20) + f"'{by_remove}'".ljust(20))


# In[ ]:


def longest_valid_parentheses(s):
    """
    Find length of longest valid parentheses substring
    LeetCode 32
    """
    stack = [-1]  # Base index
    max_len = 0

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if stack:
                max_len = max(max_len, i - stack[-1])
            else:
                stack.append(i)  # New base

    return max_len

def count_valid_substrings(s):
    """
    Count all valid parentheses substrings
    """
    count = 0
    n = len(s)

    for i in range(n):
        balance = 0
        for j in range(i, n):
            if s[j] == '(':
                balance += 1
            else:
                balance -= 1

            if balance < 0:
                break
            if balance == 0:
                count += 1

    return count

# Examples
test_cases = [
    "()",
    "(())",
    "()(()",
    "()(())",
    ")()())",
    "(()",
]

print("📊 VALID PARENTHESES ANALYSIS")
print("=" * 55)

for s in test_cases:
    longest = longest_valid_parentheses(s)
    count = count_valid_substrings(s)

    print(f"String: '{s}'")
    print(f"   Longest valid substring length: {longest}")
    print(f"   Count of valid substrings: {count}")
    print()


# In[ ]:


def parentheses_analyzer():
    """Interactive parentheses analyzer"""

    print("🔄 PARENTHESES ANALYZER")
    print("=" * 50)

    s = input("Enter string with parentheses: ")

    print("\n📋 OPTIONS:")
    print("1. Check if balanced")
    print("2. Minimum to add")
    print("3. Minimum to remove")
    print("4. Make valid (by adding)")
    print("5. Make valid (by removing)")
    print("6. Full analysis")

    choice = input("\nChoice (1-6): ")

    print("\n" + "=" * 50)
    print(f"Input: '{s}'")
    print("-" * 50)

    if choice == '1':
        result = is_balanced(s)
        print(f"Balanced: {'✅ Yes' if result else '❌ No'}")

    elif choice == '2':
        result = min_add_to_make_valid(s)
        print(f"Minimum additions needed: {result}")

    elif choice == '3':
        count, valid_str = min_remove_to_make_valid(s)
        print(f"Minimum removals needed: {count}")
        print(f"Result: '{valid_str}'")

    elif choice == '4':
        result = make_valid(s)
        print(f"Valid string: '{result}'")

    elif choice == '5':
        result = make_valid_by_removal(s)
        print(f"Valid string: '{result}'")

    elif choice == '6':
        balanced = is_balanced(s)
        min_add = min_add_to_make_valid(s)
        min_remove, removed_result = min_remove_to_make_valid(s)
        added_result = make_valid(s)

        print(f"\n📊 FULL ANALYSIS:")
        print(f"   Balanced: {'✅ Yes' if balanced else '❌ No'}")
        print(f"   Min to add: {min_add}")
        print(f"   Min to remove: {min_remove}")
        print(f"   Valid (add): '{added_result}'")
        print(f"   Valid (remove): '{removed_result}'")

    print("=" * 50)

# Run
parentheses_analyzer()


# In[ ]:


class ParenthesesAnalyzer:
    """Complete utility for parentheses operations"""

    def __init__(self, s):
        self.s = s
        self.open_char = '('
        self.close_char = ')'

    def is_balanced(self):
        """Check if balanced"""
        count = 0
        for char in self.s:
            if char == self.open_char:
                count += 1
            elif char == self.close_char:
                count -= 1
            if count < 0:
                return False
        return count == 0

    def count_unmatched(self):
        """Count unmatched parentheses"""
        open_count = 0
        close_count = 0

        for char in self.s:
            if char == self.open_char:
                open_count += 1
            elif char == self.close_char:
                if open_count > 0:
                    open_count -= 1
                else:
                    close_count += 1

        return {'unmatched_open': open_count, 'unmatched_close': close_count}

    def min_to_add(self):
        """Minimum parentheses to add"""
        counts = self.count_unmatched()
        return counts['unmatched_open'] + counts['unmatched_close']

    def min_to_remove(self):
        """Minimum parentheses to remove"""
        stack = []
        to_remove = set()

        for i, char in enumerate(self.s):
            if char == self.open_char:
                stack.append(i)
            elif char == self.close_char:
                if stack:
                    stack.pop()
                else:
                    to_remove.add(i)

        to_remove.update(stack)
        return len(to_remove)

    def make_valid_add(self):
        """Make valid by adding"""
        counts = self.count_unmatched()
        prefix = self.open_char * counts['unmatched_close']
        suffix = self.close_char * counts['unmatched_open']
        return prefix + self.s + suffix

    def make_valid_remove(self):
        """Make valid by removing"""
        stack = []
        chars = list(self.s)

        for i, char in enumerate(chars):
            if char == self.open_char:
                stack.append(i)
            elif char == self.close_char:
                if stack:
                    stack.pop()
                else:
                    chars[i] = ''

        for i in stack:
            chars[i] = ''

        return ''.join(chars)

    def get_invalid_positions(self):
        """Get positions of invalid parentheses"""
        stack = []
        invalid = []

        for i, char in enumerate(self.s):
            if char == self.open_char:
                stack.append(i)
            elif char == self.close_char:
                if stack:
                    stack.pop()
                else:
                    invalid.append(i)

        invalid.extend(stack)
        return sorted(invalid)

    def visualize(self):
        """Visual representation with invalid marked"""
        invalid = set(self.get_invalid_positions())

        result = ""
        for i, char in enumerate(self.s):
            if i in invalid:
                result += f"[{char}]"
            else:
                result += char

        return result

    def full_analysis(self):
        """Complete analysis"""
        print("\n" + "╔" + "═" * 55 + "╗")
        print("║" + "🔄 PARENTHESES ANALYSIS".center(55) + "║")
        print("╠" + "═" * 55 + "╣")

        print(f"║  📝 Input: '{self.s}'".ljust(56) + "║")
        print(f"║  📏 Length: {len(self.s)}".ljust(56) + "║")
        print("╠" + "═" * 55 + "╣")

        balanced = self.is_balanced()
        status = "✅ Yes" if balanced else "❌ No"
        print(f"║  ⚖️  Balanced: {status}".ljust(56) + "║")

        counts = self.count_unmatched()
        print(f"║  📊 Unmatched '(': {counts['unmatched_open']}".ljust(56) + "║")
        print(f"║  📊 Unmatched ')': {counts['unmatched_close']}".ljust(56) + "║")

        print("╠" + "═" * 55 + "╣")

        print(f"║  ➕ Min to add: {self.min_to_add()}".ljust(56) + "║")
        print(f"║  ➖ Min to remove: {self.min_to_remove()}".ljust(56) + "║")

        print("╠" + "═" * 55 + "╣")

        print(f"║  ✅ Valid (add): '{self.make_valid_add()}'".ljust(56) + "║")
        print(f"║  ✅ Valid (remove): '{self.make_valid_remove()}'".ljust(56) + "║")

        invalid_pos = self.get_invalid_positions()
        if invalid_pos:
            print("╠" + "═" * 55 + "╣")
            print(f"║  ❌ Invalid positions: {invalid_pos}".ljust(56) + "║")
            print(f"║  📍 Visualized: '{self.visualize()}'".ljust(56) + "║")

        print("╚" + "═" * 55 + "╝")


# Usage
test_strings = [
    "(())",
    "(()",
    "())",
    "))((",
    "(()()(",
]

for s in test_strings:
    analyzer = ParenthesesAnalyzer(s)
    analyzer.full_analysis()


# In[ ]:


s = "(()"

# Check balanced
is_bal = not any((c:=0, c:=(c+1 if x=='(' else c-1), c<0)[2] for x in s) and c==0

# Min to add (simple)
min_add = (lambda s: (lambda o,c: o+c)(*__import__('functools').reduce(
    lambda acc,x: (acc[0]+1, acc[1]) if x=='(' else 
                  (acc[0]-1, acc[1]) if acc[0]>0 else 
                  (acc[0], acc[1]+1), s, (0,0))))(s)

# Using simpler approach
def min_add_simple(s):
    o = c = 0
    for x in s:
        if x == '(': o += 1
        elif x == ')': o, c = (o-1, c) if o > 0 else (o, c+1)
    return o + c

# Make valid by adding
def quick_valid(s):
    o = c = 0
    for x in s:
        if x == '(': o += 1
        elif x == ')': o, c = (o-1, c) if o > 0 else (o, c+1)
    return '(' * c + s + ')' * o

print(f"Min to add for '{s}': {min_add_simple(s)}")
print(f"Valid: '{quick_valid(s)}'")

