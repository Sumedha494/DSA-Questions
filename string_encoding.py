#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def run_length_encode(s):
    """
    Run-Length Encoding
    "AAABBBCC" → "A3B3C2"
    Time: O(n), Space: O(n)
    """
    if not s:
        return ""

    encoded = ""
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            encoded += s[i - 1] + str(count)
            count = 1

    # Add last character group
    encoded += s[-1] + str(count)

    return encoded

# Example
test_strings = [
    "AAABBBCC",
    "AABBCC",
    "ABCD",
    "AAAA",
    "A",
]

print("🔐 RUN-LENGTH ENCODING")
print("=" * 45)

for s in test_strings:
    encoded = run_length_encode(s)
    print(f"'{s}' → '{encoded}'")


# In[ ]:


import base64

def base64_encode(s):
    """
    Base64 Encoding
    """
    # String to bytes, then base64 encode
    encoded_bytes = base64.b64encode(s.encode('utf-8'))
    return encoded_bytes.decode('utf-8')

def base64_decode(s):
    """
    Base64 Decoding
    """
    decoded_bytes = base64.b64decode(s.encode('utf-8'))
    return decoded_bytes.decode('utf-8')

# Examples
test_strings = [
    "Hello",
    "Hello World",
    "Python Programming",
    "12345",
    "Special: @#$%",
]

print("🔐 BASE64 ENCODING")
print("=" * 60)

for s in test_strings:
    encoded = base64_encode(s)
    decoded = base64_decode(encoded)

    print(f"Original: '{s}'")
    print(f"Encoded:  '{encoded}'")
    print(f"Decoded:  '{decoded}'")
    print(f"Match: {'✅' if s == decoded else '❌'}")
    print("-" * 40)


# In[ ]:


from urllib.parse import quote, unquote, urlencode

def url_encode(s):
    """URL/Percent Encoding"""
    return quote(s)

def url_decode(s):
    """URL/Percent Decoding"""
    return unquote(s)

# Examples
test_strings = [
    "Hello World",
    "Python & Java",
    "100% Complete",
    "user@email.com",
    "price=$50",
    "search?q=python",
]

print("🔐 URL ENCODING")
print("=" * 55)

for s in test_strings:
    encoded = url_encode(s)
    decoded = url_decode(encoded)

    print(f"Original: '{s}'")
    print(f"Encoded:  '{encoded}'")
    print(f"Decoded:  '{decoded}'")
    print("-" * 40)


# In[ ]:


def demonstrate_character_encoding(s):
    """
    Demonstrate different character encodings
    """
    print(f"🔤 CHARACTER ENCODING: '{s}'")
    print("=" * 55)

    encodings = ['utf-8', 'utf-16', 'ascii', 'latin-1']

    for encoding in encodings:
        try:
            # Encode to bytes
            encoded = s.encode(encoding)

            # Decode back
            decoded = encoded.decode(encoding)

            print(f"\n📌 {encoding.upper()}:")
            print(f"   Bytes: {encoded}")
            print(f"   Hex:   {encoded.hex()}")
            print(f"   Length: {len(encoded)} bytes")

        except UnicodeEncodeError as e:
            print(f"\n📌 {encoding.upper()}: ❌ Cannot encode ({e})")

    print()

# Examples
demonstrate_character_encoding("Hello")
demonstrate_character_encoding("नमस्ते")  # Hindi
demonstrate_character_encoding("你好")    # Chinese


# In[ ]:


def caesar_encode(text, shift=3):
    """
    Caesar Cipher Encoding
    Shift each letter by 'shift' positions
    """
    result = ""

    for char in text:
        if char.isalpha():
            # Determine case
            base = ord('A') if char.isupper() else ord('a')

            # Shift character
            shifted = (ord(char) - base + shift) % 26 + base
            result += chr(shifted)
        else:
            result += char

    return result

def caesar_decode(text, shift=3):
    """
    Caesar Cipher Decoding
    """
    return caesar_encode(text, -shift)

# Examples
text = "Hello World"
shift = 3

print("🔐 CAESAR CIPHER")
print("=" * 45)
print(f"Original: '{text}'")
print(f"Shift: {shift}")

encoded = caesar_encode(text, shift)
decoded = caesar_decode(encoded, shift)

print(f"Encoded: '{encoded}'")
print(f"Decoded: '{decoded}'")
print()

# Show all shifts
print("📊 ALL SHIFTS:")
print("-" * 30)
for s in range(1, 6):
    print(f"Shift {s}: '{caesar_encode(text, s)}'")


# In[ ]:


def hex_encode(s):
    """Convert string to hexadecimal"""
    return s.encode('utf-8').hex()

def hex_decode(hex_str):
    """Convert hexadecimal to string"""
    return bytes.fromhex(hex_str).decode('utf-8')

# Examples
test_strings = [
    "Hello",
    "Python",
    "123",
    "ABC",
]

print("🔐 HEX ENCODING")
print("=" * 50)

for s in test_strings:
    encoded = hex_encode(s)
    decoded = hex_decode(encoded)

    print(f"Original: '{s}'")
    print(f"Hex:      '{encoded}'")
    print(f"Decoded:  '{decoded}'")
    print()

# Show character-by-character
print("📊 CHARACTER HEX VALUES:")
print("-" * 30)
text = "Hello"
for char in text:
    hex_val = ord(char)
    print(f"'{char}' → {hex_val} → 0x{hex_val:02x}")


# In[ ]:


def binary_encode(s):
    """Convert string to binary"""
    return ' '.join(format(ord(char), '08b') for char in s)

def binary_decode(binary_str):
    """Convert binary to string"""
    binary_values = binary_str.split()
    return ''.join(chr(int(b, 2)) for b in binary_values)

# Examples
test_strings = [
    "Hi",
    "AB",
    "123",
]

print("🔐 BINARY ENCODING")
print("=" * 55)

for s in test_strings:
    encoded = binary_encode(s)
    decoded = binary_decode(encoded)

    print(f"Original: '{s}'")
    print(f"Binary:   '{encoded}'")
    print(f"Decoded:  '{decoded}'")
    print("-" * 40)

# Detailed view
print("\n📊 CHARACTER BINARY VALUES:")
print("-" * 45)
text = "Hello"
for char in text:
    binary = format(ord(char), '08b')
    print(f"'{char}' → ASCII {ord(char):3} → Binary {binary}")


# In[ ]:


import codecs

def rot13_encode(s):
    """
    ROT13 - Rotate by 13 positions
    Special case of Caesar cipher
    Encoding = Decoding (symmetric)
    """
    return codecs.encode(s, 'rot_13')

def rot13_manual(s):
    """Manual ROT13 implementation"""
    result = ""

    for char in s:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - base + 13) % 26 + base)
        else:
            result += char

    return result

# Examples
text = "Hello World"

print("🔐 ROT13 ENCODING")
print("=" * 45)
print(f"Original:     '{text}'")

encoded = rot13_encode(text)
print(f"Encoded:      '{encoded}'")

# ROT13 is symmetric - encode again to decode
decoded = rot13_encode(encoded)
print(f"Double ROT13: '{decoded}'")

print()
print("📊 ROT13 is symmetric: encode(encode(x)) = x")
print("-" * 45)
print(f"'ABC' → '{rot13_encode('ABC')}' → '{rot13_encode(rot13_encode('ABC'))}'")


# In[ ]:


MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '0': '-----', '1': '.----', '2': '..---', 
    '3': '...--', '4': '....-', '5': '.....', '6': '-....',
    '7': '--...', '8': '---..', '9': '----.', ' ': '/'
}

# Reverse mapping for decoding
MORSE_DECODE = {v: k for k, v in MORSE_CODE.items()}

def morse_encode(text):
    """Convert text to Morse code"""
    return ' '.join(MORSE_CODE.get(char.upper(), '') for char in text)

def morse_decode(morse):
    """Convert Morse code to text"""
    return ''.join(MORSE_DECODE.get(code, '') for code in morse.split())

# Examples
text = "HELLO"

print("🔐 MORSE CODE ENCODING")
print("=" * 55)
print(f"Original: '{text}'")

encoded = morse_encode(text)
decoded = morse_decode(encoded)

print(f"Morse:    '{encoded}'")
print(f"Decoded:  '{decoded}'")

print("\n📊 CHARACTER MORSE VALUES:")
print("-" * 35)
for char in text:
    print(f"'{char}' → {MORSE_CODE[char]}")

# Full example
print("\n📝 FULL EXAMPLE:")
text2 = "SOS"
encoded2 = morse_encode(text2)
print(f"'{text2}' → '{encoded2}'")


# In[ ]:


def xor_encode(text, key):
    """
    XOR Encoding with key
    Simple symmetric encryption
    """
    encoded = []

    for i, char in enumerate(text):
        # XOR with corresponding key character
        key_char = key[i % len(key)]
        encoded_char = chr(ord(char) ^ ord(key_char))
        encoded.append(encoded_char)

    return ''.join(encoded)

def xor_decode(encoded_text, key):
    """
    XOR Decoding - same as encoding (symmetric)
    """
    return xor_encode(encoded_text, key)

# Example
text = "Hello"
key = "KEY"

print("🔐 XOR ENCODING")
print("=" * 50)
print(f"Text: '{text}'")
print(f"Key:  '{key}'")

encoded = xor_encode(text, key)
decoded = xor_decode(encoded, key)

print(f"\nEncoded (hex): {encoded.encode().hex()}")
print(f"Decoded: '{decoded}'")

print("\n📊 STEP-BY-STEP XOR:")
print("-" * 45)
for i, char in enumerate(text):
    key_char = key[i % len(key)]
    result = ord(char) ^ ord(key_char)
    print(f"'{char}' XOR '{key_char}' = {ord(char)} XOR {ord(key_char)} = {result}")


# In[ ]:


import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    """Build Huffman tree from text"""
    # Count frequency
    freq = Counter(text)

    # Create priority queue
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)

    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)

    return heap[0] if heap else None

def generate_codes(node, code="", codes=None):
    """Generate Huffman codes"""
    if codes is None:
        codes = {}

    if node:
        if node.char is not None:
            codes[node.char] = code if code else "0"
        generate_codes(node.left, code + "0", codes)
        generate_codes(node.right, code + "1", codes)

    return codes

def huffman_encode(text):
    """Huffman encoding"""
    if not text:
        return "", {}

    tree = build_huffman_tree(text)
    codes = generate_codes(tree)
    encoded = ''.join(codes[char] for char in text)

    return encoded, codes

def huffman_decode(encoded, codes):
    """Huffman decoding"""
    reverse_codes = {v: k for k, v in codes.items()}
    decoded = ""
    current = ""

    for bit in encoded:
        current += bit
        if current in reverse_codes:
            decoded += reverse_codes[current]
            current = ""

    return decoded

# Example
text = "ABRACADABRA"

print("🔐 HUFFMAN ENCODING")
print("=" * 55)
print(f"Text: '{text}'")

encoded, codes = huffman_encode(text)
decoded = huffman_decode(encoded, codes)

print(f"\n📊 Character Frequencies:")
for char, count in Counter(text).items():
    print(f"   '{char}': {count}")

print(f"\n📊 Huffman Codes:")
for char, code in sorted(codes.items()):
    print(f"   '{char}': {code}")

print(f"\nEncoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"\nOriginal bits: {len(text) * 8}")
print(f"Encoded bits:  {len(encoded)}")
print(f"Compression:   {100 - len(encoded)/(len(text)*8)*100:.1f}%")


# In[ ]:


import base64
import codecs
from urllib.parse import quote, unquote

class StringEncoder:
    """Complete string encoding utility"""

    def __init__(self, text):
        self.text = text

    # ====== RUN-LENGTH ENCODING ======
    def rle_encode(self):
        """Run-Length Encoding"""
        if not self.text:
            return ""

        result = ""
        count = 1

        for i in range(1, len(self.text)):
            if self.text[i] == self.text[i-1]:
                count += 1
            else:
                result += self.text[i-1] + str(count)
                count = 1

        result += self.text[-1] + str(count)
        return result

    # ====== BASE64 ======
    def base64_encode(self):
        """Base64 Encoding"""
        return base64.b64encode(self.text.encode()).decode()

    # ====== URL ENCODING ======
    def url_encode(self):
        """URL/Percent Encoding"""
        return quote(self.text)

    # ====== HEX ======
    def hex_encode(self):
        """Hexadecimal Encoding"""
        return self.text.encode().hex()

    # ====== BINARY ======
    def binary_encode(self):
        """Binary Encoding"""
        return ' '.join(format(ord(c), '08b') for c in self.text)

    # ====== CAESAR CIPHER ======
    def caesar_encode(self, shift=3):
        """Caesar Cipher"""
        result = ""
        for char in self.text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - base + shift) % 26 + base)
            else:
                result += char
        return result

    # ====== ROT13 ======
    def rot13_encode(self):
        """ROT13 Encoding"""
        return codecs.encode(self.text, 'rot_13')

    # ====== REVERSE ======
    def reverse_encode(self):
        """Simple Reverse"""
        return self.text[::-1]

    # ====== ASCII VALUES ======
    def ascii_encode(self):
        """Convert to ASCII values"""
        return '-'.join(str(ord(c)) for c in self.text)

    def show_all_encodings(self):
        """Display all encodings"""
        print("\n" + "╔" + "═" * 60 + "╗")
        print("║" + "🔐 ALL STRING ENCODINGS".center(60) + "║")
        print("╠" + "═" * 60 + "╣")

        text_display = self.text if len(self.text) <= 40 else self.text[:37] + "..."
        print(f"║  📝 Original: '{text_display}'".ljust(61) + "║")
        print(f"║  📏 Length: {len(self.text)} characters".ljust(61) + "║")
        print("╠" + "═" * 60 + "╣")

        encodings = [
            ("RLE", self.rle_encode()),
            ("Base64", self.base64_encode()),
            ("URL", self.url_encode()),
            ("Hex", self.hex_encode()),
            ("Binary", self.binary_encode()[:40] + "..."),
            ("Caesar (3)", self.caesar_encode(3)),
            ("ROT13", self.rot13_encode()),
            ("Reverse", self.reverse_encode()),
            ("ASCII", self.ascii_encode()),
        ]

        for name, encoded in encodings:
            display = encoded if len(encoded) <= 40 else encoded[:37] + "..."
            print(f"║  📌 {name}:".ljust(61) + "║")
            print(f"║     {display}".ljust(61) + "║")

        print("╚" + "═" * 60 + "╝")


# Usage
text = "Hello World"
encoder = StringEncoder(text)
encoder.show_all_encodings()


# In[ ]:


def encoding_tool():
    """Interactive encoding tool"""

    print("🔐 STRING ENCODING TOOL")
    print("=" * 50)

    text = input("Enter text to encode: ")

    print("\n📋 SELECT ENCODING:")
    print("1. Run-Length Encoding (RLE)")
    print("2. Base64")
    print("3. URL Encoding")
    print("4. Hex")
    print("5. Binary")
    print("6. Caesar Cipher")
    print("7. ROT13")
    print("8. Morse Code")
    print("9. Reverse")
    print("10. Show All")

    choice = input("\nChoice (1-10): ")

    print("\n" + "=" * 50)
    print(f"Original: '{text}'")
    print("-" * 50)

    encoder = StringEncoder(text)

    if choice == '1':
        print(f"RLE: {encoder.rle_encode()}")
    elif choice == '2':
        print(f"Base64: {encoder.base64_encode()}")
    elif choice == '3':
        print(f"URL: {encoder.url_encode()}")
    elif choice == '4':
        print(f"Hex: {encoder.hex_encode()}")
    elif choice == '5':
        print(f"Binary: {encoder.binary_encode()}")
    elif choice == '6':
        shift = int(input("Enter shift (1-25): ") or "3")
        print(f"Caesar ({shift}): {encoder.caesar_encode(shift)}")
    elif choice == '7':
        print(f"ROT13: {encoder.rot13_encode()}")
    elif choice == '8':
        print(f"Morse: {morse_encode(text)}")
    elif choice == '9':
        print(f"Reverse: {encoder.reverse_encode()}")
    elif choice == '10':
        encoder.show_all_encodings()

    print("=" * 50)

# Run
encoding_tool()


# In[ ]:


def compare_encodings(text):
    """Compare all encoding methods"""

    encodings = {
        'Original': text,
        'RLE': StringEncoder(text).rle_encode(),
        'Base64': base64.b64encode(text.encode()).decode(),
        'URL': quote(text),
        'Hex': text.encode().hex(),
        'Caesar-3': StringEncoder(text).caesar_encode(3),
        'ROT13': codecs.encode(text, 'rot_13'),
        'Reverse': text[::-1],
    }

    print("\n📊 ENCODING COMPARISON")
    print("=" * 70)
    print(f"{'Method':<12}│{'Encoded':<40}│{'Length':<8}│{'Ratio'}")
    print("-" * 70)

    orig_len = len(text)

    for name, encoded in encodings.items():
        display = encoded[:35] + "..." if len(encoded) > 35 else encoded
        ratio = len(encoded) / orig_len if orig_len > 0 else 0
        print(f"{name:<12}│{display:<40}│{len(encoded):<8}│{ratio:.2f}x")

    print("=" * 70)

# Example
compare_encodings("Hello World")
compare_encodings("AAABBBCCC")

