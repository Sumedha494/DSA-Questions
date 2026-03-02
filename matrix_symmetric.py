#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def is_symmetric(matrix):
    """
    Check if matrix is symmetric
    Time: O(n²), Space: O(1)
    """
    n = len(matrix)

    # Must be square matrix
    for row in matrix:
        if len(row) != n:
            return False

    # Check A[i][j] == A[j][i]
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle
            if matrix[i][j] != matrix[j][i]:
                return False

    return True

# Examples
symmetric_matrix = [
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 6]
]

non_symmetric_matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print("🔢 CHECK SYMMETRIC MATRIX")
print("=" * 45)

print("Matrix 1:")
for row in symmetric_matrix:
    print(f"   {row}")
print(f"Symmetric: {'✅ Yes' if is_symmetric(symmetric_matrix) else '❌ No'}")

print("\nMatrix 2:")
for row in non_symmetric_matrix:
    print(f"   {row}")
print(f"Symmetric: {'✅ Yes' if is_symmetric(non_symmetric_matrix) else '❌ No'}")


# In[ ]:


def get_transpose(matrix):
    """Get transpose of matrix"""
    n = len(matrix)
    m = len(matrix[0])

    transpose = [[0] * n for _ in range(m)]

    for i in range(n):
        for j in range(m):
            transpose[j][i] = matrix[i][j]

    return transpose

def is_symmetric_transpose(matrix):
    """Check symmetric using transpose comparison"""
    n = len(matrix)

    # Must be square
    if any(len(row) != n for row in matrix):
        return False

    transpose = get_transpose(matrix)

    return matrix == transpose

# One-liner using zip
def is_symmetric_zip(matrix):
    return matrix == [list(row) for row in zip(*matrix)]

# Example
matrix = [
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 6]
]

print("🔢 SYMMETRIC CHECK USING TRANSPOSE")
print("=" * 45)

print("Original Matrix:")
for row in matrix:
    print(f"   {row}")

transpose = get_transpose(matrix)
print("\nTranspose:")
for row in transpose:
    print(f"   {row}")

print(f"\nOriginal == Transpose: {'✅ Yes' if matrix == transpose else '❌ No'}")
print(f"Symmetric: {'✅ Yes' if is_symmetric_transpose(matrix) else '❌ No'}")


# In[ ]:


def check_symmetric_visualized(matrix):
    """
    Visualize symmetric check step by step
    """
    print("🔢 SYMMETRIC MATRIX CHECK - VISUALIZATION")
    print("=" * 55)

    n = len(matrix)

    # Print matrix
    print("Matrix:")
    for i, row in enumerate(matrix):
        print(f"   Row {i}: {row}")

    print("\n" + "-" * 55)
    print("Checking A[i][j] == A[j][i] for upper triangle:")
    print("-" * 55)

    is_sym = True
    comparisons = []

    for i in range(n):
        for j in range(i + 1, n):
            val_ij = matrix[i][j]
            val_ji = matrix[j][i]

            match = val_ij == val_ji
            status = "✅" if match else "❌"

            comparisons.append({
                'i': i, 'j': j,
                'A[i][j]': val_ij,
                'A[j][i]': val_ji,
                'match': match
            })

            print(f"   A[{i}][{j}] = {val_ij}  vs  A[{j}][{i}] = {val_ji}  {status}")

            if not match:
                is_sym = False

    print("\n" + "-" * 55)
    print(f"📊 RESULT: {'✅ SYMMETRIC' if is_sym else '❌ NOT SYMMETRIC'}")

    if not is_sym:
        print("\n⚠️ Mismatches found at:")
        for c in comparisons:
            if not c['match']:
                print(f"   Position ({c['i']},{c['j']}) and ({c['j']},{c['i']})")

    return is_sym

# Examples
matrix1 = [
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 6]
]

matrix2 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

check_symmetric_visualized(matrix1)
print("\n")
check_symmetric_visualized(matrix2)


# In[ ]:


def make_symmetric_upper(matrix):
    """
    Make symmetric by copying upper triangle to lower
    """
    n = len(matrix)
    result = [row[:] for row in matrix]  # Deep copy

    for i in range(n):
        for j in range(i + 1, n):
            result[j][i] = result[i][j]

    return result

def make_symmetric_lower(matrix):
    """
    Make symmetric by copying lower triangle to upper
    """
    n = len(matrix)
    result = [row[:] for row in matrix]

    for i in range(n):
        for j in range(i + 1, n):
            result[i][j] = result[j][i]

    return result

def make_symmetric_average(matrix):
    """
    Make symmetric by averaging A[i][j] and A[j][i]
    """
    n = len(matrix)
    result = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            result[i][j] = (matrix[i][j] + matrix[j][i]) / 2

    return result

# Example
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print("🔢 MAKE MATRIX SYMMETRIC")
print("=" * 45)

print("Original Matrix:")
for row in matrix:
    print(f"   {row}")

print("\nSymmetric (Upper → Lower):")
sym_upper = make_symmetric_upper(matrix)
for row in sym_upper:
    print(f"   {row}")

print("\nSymmetric (Lower → Upper):")
sym_lower = make_symmetric_lower(matrix)
for row in sym_lower:
    print(f"   {row}")

print("\nSymmetric (Average):")
sym_avg = make_symmetric_average(matrix)
for row in sym_avg:
    print(f"   {row}")


# In[ ]:


import numpy as np

def symmetric_operations_numpy(matrix):
    """
    Symmetric matrix operations using NumPy
    """
    A = np.array(matrix)

    print("🔢 NUMPY SYMMETRIC OPERATIONS")
    print("=" * 50)

    print("Original Matrix:")
    print(A)

    # Check symmetric
    is_sym = np.allclose(A, A.T)
    print(f"\nIs Symmetric: {'✅ Yes' if is_sym else '❌ No'}")

    # Transpose
    print(f"\nTranspose (A.T):")
    print(A.T)

    # Make symmetric using (A + A.T) / 2
    symmetric = (A + A.T) / 2
    print(f"\nSymmetric Part (A + A.T) / 2:")
    print(symmetric)

    # Skew-symmetric part
    skew_symmetric = (A - A.T) / 2
    print(f"\nSkew-Symmetric Part (A - A.T) / 2:")
    print(skew_symmetric)

    # Verify: A = symmetric + skew-symmetric
    reconstructed = symmetric + skew_symmetric
    print(f"\nVerify: Symmetric + Skew = Original")
    print(f"Match: {np.allclose(A, reconstructed)}")

    return is_sym

# Example
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

symmetric_operations_numpy(matrix)


# In[ ]:


def is_skew_symmetric(matrix):
    """
    Check if matrix is skew-symmetric: A = -Aᵀ
    A[i][j] = -A[j][i] for all i, j
    Diagonal elements must be 0
    """
    n = len(matrix)

    for i in range(n):
        # Diagonal must be 0
        if matrix[i][i] != 0:
            return False

        for j in range(i + 1, n):
            if matrix[i][j] != -matrix[j][i]:
                return False

    return True

def make_skew_symmetric(matrix):
    """
    Make skew-symmetric using (A - Aᵀ) / 2
    """
    n = len(matrix)
    result = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            result[i][j] = (matrix[i][j] - matrix[j][i]) / 2

    return result

# Examples
skew_matrix = [
    [0, 2, -1],
    [-2, 0, 3],
    [1, -3, 0]
]

normal_matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print("🔢 SKEW-SYMMETRIC MATRIX")
print("=" * 50)

print("Skew-Symmetric Matrix:")
for row in skew_matrix:
    print(f"   {row}")
print(f"Is Skew-Symmetric: {'✅ Yes' if is_skew_symmetric(skew_matrix) else '❌ No'}")

print("\nNormal Matrix:")
for row in normal_matrix:
    print(f"   {row}")
print(f"Is Skew-Symmetric: {'✅ Yes' if is_skew_symmetric(normal_matrix) else '❌ No'}")

print("\nSkew-Symmetric Part of Normal Matrix:")
skew_part = make_skew_symmetric(normal_matrix)
for row in skew_part:
    print(f"   {row}")


# In[ ]:


import numpy as np

def symmetric_properties(matrix):
    """
    Demonstrate properties of symmetric matrices
    """
    A = np.array(matrix, dtype=float)

    print("🔢 SYMMETRIC MATRIX PROPERTIES")
    print("=" * 55)

    print("Matrix A:")
    print(A)

    # Check if symmetric
    is_sym = np.allclose(A, A.T)
    print(f"\n1️⃣ Is Symmetric (A = Aᵀ): {'✅ Yes' if is_sym else '❌ No'}")

    if not is_sym:
        print("   Making symmetric for demonstration...")
        A = (A + A.T) / 2
        print("   Symmetric A:")
        print(A)

    # Property 2: Eigenvalues are real
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\n2️⃣ Eigenvalues (all real for symmetric):")
    print(f"   {eigenvalues}")
    print(f"   All real: {'✅ Yes' if np.allclose(eigenvalues.imag, 0) else '❌ No'}")

    # Property 3: Eigenvectors are orthogonal
    print(f"\n3️⃣ Eigenvectors are orthogonal:")
    print(f"   V.T @ V (should be identity):")
    VTV = eigenvectors.T @ eigenvectors
    print(np.round(VTV, 4))

    # Property 4: A² is also symmetric
    A_squared = A @ A
    print(f"\n4️⃣ A² is also symmetric:")
    print(f"   A²:")
    print(A_squared)
    print(f"   A² symmetric: {'✅ Yes' if np.allclose(A_squared, A_squared.T) else '❌ No'}")

    # Property 5: Determinant
    det = np.linalg.det(A)
    print(f"\n5️⃣ Determinant: {det:.4f}")

    # Property 6: Trace (sum of diagonal)
    trace = np.trace(A)
    print(f"\n6️⃣ Trace (sum of eigenvalues): {trace:.4f}")
    print(f"   Sum of eigenvalues: {sum(eigenvalues.real):.4f}")

    return A

# Example
matrix = [
    [4, 2, 2],
    [2, 5, 1],
    [2, 1, 6]
]

symmetric_properties(matrix)


# In[ ]:


import random

def generate_symmetric_matrix(n, min_val=0, max_val=10):
    """
    Generate random symmetric matrix
    """
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            val = random.randint(min_val, max_val)
            matrix[i][j] = val
            matrix[j][i] = val

    return matrix

def generate_positive_definite(n):
    """
    Generate symmetric positive definite matrix
    A = B @ B.T where B is random
    """
    import numpy as np

    B = np.random.rand(n, n)
    A = B @ B.T

    return A.tolist()

# Examples
print("🔢 GENERATE SYMMETRIC MATRIX")
print("=" * 45)

print("\n1. Random Symmetric (3x3):")
sym1 = generate_symmetric_matrix(3)
for row in sym1:
    print(f"   {row}")

print("\n2. Random Symmetric (4x4):")
sym2 = generate_symmetric_matrix(4, -5, 5)
for row in sym2:
    print(f"   {row}")

print("\n3. Positive Definite (3x3):")
pd = generate_positive_definite(3)
for row in pd:
    print(f"   {[round(x, 2) for x in row]}")

# Verify
print("\n✅ Verification:")
print(f"   Matrix 1 symmetric: {is_symmetric(sym1)}")
print(f"   Matrix 2 symmetric: {is_symmetric(sym2)}")


# In[ ]:


def symmetric_matrix_tool():
    """Interactive symmetric matrix tool"""

    print("🔢 SYMMETRIC MATRIX TOOL")
    print("=" * 50)

    print("\n📋 OPTIONS:")
    print("1. Check if matrix is symmetric")
    print("2. Make matrix symmetric")
    print("3. Generate symmetric matrix")
    print("4. Check skew-symmetric")
    print("5. Decompose into symmetric + skew-symmetric")
    print("6. Full analysis")

    choice = input("\nChoice (1-6): ")

    if choice == '3':
        n = int(input("Enter matrix size: "))
        matrix = generate_symmetric_matrix(n)
        print(f"\nGenerated {n}x{n} Symmetric Matrix:")
        for row in matrix:
            print(f"   {row}")
        return

    # Get matrix input
    n = int(input("Enter matrix size: "))
    print(f"Enter {n}x{n} matrix (row by row, space separated):")

    matrix = []
    for i in range(n):
        row = list(map(float, input(f"Row {i}: ").split()))
        matrix.append(row)

    print("\n" + "=" * 50)
    print("Your Matrix:")
    for row in matrix:
        print(f"   {row}")
    print("-" * 50)

    if choice == '1':
        result = is_symmetric(matrix)
        print(f"Symmetric: {'✅ Yes' if result else '❌ No'}")

    elif choice == '2':
        print("\nSymmetric (Upper → Lower):")
        sym = make_symmetric_upper(matrix)
        for row in sym:
            print(f"   {row}")

        print("\nSymmetric (Average):")
        sym_avg = make_symmetric_average(matrix)
        for row in sym_avg:
            print(f"   {row}")

    elif choice == '4':
        result = is_skew_symmetric(matrix)
        print(f"Skew-Symmetric: {'✅ Yes' if result else '❌ No'}")

    elif choice == '5':
        print("\nSymmetric Part (A + Aᵀ)/2:")
        sym_part = make_symmetric_average(matrix)
        for row in sym_part:
            print(f"   {row}")

        print("\nSkew-Symmetric Part (A - Aᵀ)/2:")
        skew_part = make_skew_symmetric(matrix)
        for row in skew_part:
            print(f"   {row}")

    elif choice == '6':
        print(f"\n📊 FULL ANALYSIS:")
        print(f"   Is Symmetric: {'✅' if is_symmetric(matrix) else '❌'}")
        print(f"   Is Skew-Symmetric: {'✅' if is_skew_symmetric(matrix) else '❌'}")
        print(f"   Size: {n}x{n}")

        # Trace
        trace = sum(matrix[i][i] for i in range(n))
        print(f"   Trace: {trace}")

        # Check diagonal
        diagonal = [matrix[i][i] for i in range(n)]
        print(f"   Diagonal: {diagonal}")

# Run
symmetric_matrix_tool()


# In[ ]:


import numpy as np

class SymmetricMatrix:
    """Complete utility for symmetric matrix operations"""

    def __init__(self, matrix):
        self.matrix = [row[:] for row in matrix]
        self.n = len(matrix)
        self.np_matrix = np.array(matrix, dtype=float)

    def is_square(self):
        """Check if matrix is square"""
        return all(len(row) == self.n for row in self.matrix)

    def is_symmetric(self):
        """Check if matrix is symmetric"""
        if not self.is_square():
            return False

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.matrix[i][j] != self.matrix[j][i]:
                    return False
        return True

    def is_skew_symmetric(self):
        """Check if matrix is skew-symmetric"""
        for i in range(self.n):
            if self.matrix[i][i] != 0:
                return False
            for j in range(i + 1, self.n):
                if self.matrix[i][j] != -self.matrix[j][i]:
                    return False
        return True

    def get_transpose(self):
        """Get transpose of matrix"""
        return [[self.matrix[j][i] for j in range(self.n)] for i in range(self.n)]

    def get_symmetric_part(self):
        """Get symmetric part: (A + Aᵀ) / 2"""
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                result[i][j] = (self.matrix[i][j] + self.matrix[j][i]) / 2
        return result

    def get_skew_symmetric_part(self):
        """Get skew-symmetric part: (A - Aᵀ) / 2"""
        result = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                result[i][j] = (self.matrix[i][j] - self.matrix[j][i]) / 2
        return result

    def make_symmetric(self, method='upper'):
        """Make matrix symmetric"""
        result = [row[:] for row in self.matrix]

        if method == 'upper':
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    result[j][i] = result[i][j]
        elif method == 'lower':
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    result[i][j] = result[j][i]
        elif method == 'average':
            result = self.get_symmetric_part()

        return result

    def get_diagonal(self):
        """Get diagonal elements"""
        return [self.matrix[i][i] for i in range(self.n)]

    def get_trace(self):
        """Get trace (sum of diagonal)"""
        return sum(self.get_diagonal())

    def get_eigenvalues(self):
        """Get eigenvalues (real for symmetric)"""
        if self.is_symmetric():
            return np.linalg.eigvalsh(self.np_matrix).tolist()
        return np.linalg.eigvals(self.np_matrix).tolist()

    def is_positive_definite(self):
        """Check if positive definite (all eigenvalues > 0)"""
        if not self.is_symmetric():
            return False
        eigenvalues = self.get_eigenvalues()
        return all(ev > 0 for ev in eigenvalues)

    def is_positive_semidefinite(self):
        """Check if positive semi-definite (all eigenvalues >= 0)"""
        if not self.is_symmetric():
            return False
        eigenvalues = self.get_eigenvalues()
        return all(ev >= -1e-10 for ev in eigenvalues)

    def get_mismatches(self):
        """Get positions where A[i][j] != A[j][i]"""
        mismatches = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.matrix[i][j] != self.matrix[j][i]:
                    mismatches.append({
                        'pos': (i, j),
                        'A[i][j]': self.matrix[i][j],
                        'A[j][i]': self.matrix[j][i],
                        'diff': abs(self.matrix[i][j] - self.matrix[j][i])
                    })
        return mismatches

    def full_analysis(self):
        """Complete analysis"""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + "🔢 SYMMETRIC MATRIX ANALYSIS".center(58) + "║")
        print("╠" + "═" * 58 + "╣")

        # Print matrix
        print("║  Matrix:".ljust(59) + "║")
        for row in self.matrix:
            row_str = "   " + str([round(x, 2) if isinstance(x, float) else x for x in row])
            print(f"║  {row_str}".ljust(59) + "║")

        print("╠" + "═" * 58 + "╣")

        # Properties
        is_sym = self.is_symmetric()
        is_skew = self.is_skew_symmetric()

        print(f"║  📏 Size: {self.n}x{self.n}".ljust(59) + "║")
        print(f"║  ✅ Symmetric: {'✅ Yes' if is_sym else '❌ No'}".ljust(59) + "║")
        print(f"║  ✅ Skew-Symmetric: {'✅ Yes' if is_skew else '❌ No'}".ljust(59) + "║")
        print(f"║  📊 Trace: {self.get_trace()}".ljust(59) + "║")
        print(f"║  📊 Diagonal: {self.get_diagonal()}".ljust(59) + "║")

        # Eigenvalues
        try:
            eigenvalues = self.get_eigenvalues()
            ev_str = [round(ev, 4) if isinstance(ev, float) else round(ev.real, 4) for ev in eigenvalues]
            print(f"║  📊 Eigenvalues: {ev_str}".ljust(59) + "║")

            if is_sym:
                print(f"║  ✅ Positive Definite: {'✅ Yes' if self.is_positive_definite() else '❌ No'}".ljust(59) + "║")
        except:
            pass

        # Mismatches
        mismatches = self.get_mismatches()
        if mismatches:
            print("╠" + "═" * 58 + "╣")
            print(f"║  ⚠️ Mismatches ({len(mismatches)}):".ljust(59) + "║")
            for m in mismatches[:3]:
                print(f"║     ({m['pos'][0]},{m['pos'][1]}): {m['A[i][j]']} vs {m['A[j][i]']}".ljust(59) + "║")

        # Symmetric part
        if not is_sym:
            print("╠" + "═" * 58 + "╣")
            print("║  🔧 Symmetric Part:".ljust(59) + "║")
            sym_part = self.get_symmetric_part()
            for row in sym_part:
                row_str = "   " + str([round(x, 2) for x in row])
                print(f"║  {row_str}".ljust(59) + "║")

        print("╚" + "═" * 58 + "╝")


# Usage Examples
print("📊 SYMMETRIC MATRIX:")
sym_matrix = [
    [4, 2, 1],
    [2, 5, 3],
    [1, 3, 6]
]
analyzer1 = SymmetricMatrix(sym_matrix)
analyzer1.full_analysis()

print("\n📊 NON-SYMMETRIC MATRIX:")
non_sym_matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
analyzer2 = SymmetricMatrix(non_sym_matrix)
analyzer2.full_analysis()


# In[ ]:


matrix = [[1, 2, 3], [2, 5, 6], [3, 6, 9]]

# Check symmetric
is_sym = all(matrix[i][j] == matrix[j][i] for i in range(len(matrix)) for j in range(len(matrix)))
print(f"Symmetric: {is_sym}")

# Check symmetric using transpose
is_sym_v2 = matrix == [list(row) for row in zip(*matrix)]
print(f"Symmetric (zip): {is_sym_v2}")

# Get transpose
transpose = [list(row) for row in zip(*matrix)]
print(f"Transpose: {transpose}")

# Get diagonal
diagonal = [matrix[i][i] for i in range(len(matrix))]
print(f"Diagonal: {diagonal}")

# Get trace
trace = sum(matrix[i][i] for i in range(len(matrix)))
print(f"Trace: {trace}")

# Make symmetric (upper to lower)
n = len(matrix)
sym = [[matrix[min(i,j)][max(i,j)] for j in range(n)] for i in range(n)]
print(f"Made symmetric: {sym}")

# Count mismatches
mismatches = sum(1 for i in range(n) for j in range(i+1, n) if matrix[i][j] != matrix[j][i])
print(f"Mismatches: {mismatches}")

