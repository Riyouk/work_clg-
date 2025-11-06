# =============================================
# 🧠 COMPLETE NUMPY + LAMBDA + MAP/FILTER/REDUCE + MELT REFERENCE
# =============================================

import numpy as np
import pandas as pd
from functools import reduce
import math
import timeit

print("\n============================")
print("📘 1️⃣ ARRAY CREATION")
print("============================")
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
arr3 = np.arange(0, 10, 2)
arr4 = np.zeros((2, 3))
arr5 = np.ones((3, 3))
arr6 = np.eye(3)
print(arr1, arr2, arr3, arr4, arr5, arr6, sep="\n\n")

# ---------------------------------
print("\n============================")
print("📗 2️⃣ ARRAY ATTRIBUTES")
print("============================")
print("Dimensions:", arr2.ndim)
print("Shape:", arr2.shape)
print("Size:", arr2.size)
print("Data Type:", arr2.dtype)
print("Item Size (bytes):", arr2.itemsize)

# ---------------------------------
print("\n============================")
print("📘 3️⃣ RESHAPING & FLATTENING")
print("============================")
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
flattened = reshaped.flatten()
print("Reshaped (3x4):\n", reshaped)
print("Flattened:", flattened)

# ---------------------------------
print("\n============================")
print("📗 4️⃣ INDEXING & SLICING")
print("============================")
a = np.array([10, 20, 30, 40, 50])
print("a[1:4] =", a[1:4])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("b[:,1] =", b[:,1])
print("b[0:2,1:3] =\n", b[0:2, 1:3])

# ---------------------------------
print("\n============================")
print("📘 5️⃣ MATHEMATICAL OPERATIONS")
print("============================")
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("Add:", np.add(x, y))
print("Subtract:", np.subtract(x, y))
print("Multiply:", np.multiply(x, y))
print("Divide:", np.divide(x, y))
print("Power:", np.power(x, 2))
print("Modulus:", np.mod(y, x))

# ---------------------------------
print("\n============================")
print("📗 6️⃣ STATISTICAL & MATH FUNCTIONS")
print("============================")
data = np.array([10, 20, 30, 40, 50])
print("Sum:", np.sum(data))
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Std:", np.std(data))
print("Variance:", np.var(data))
print("Min/Max:", np.min(data), np.max(data))
print("Argmin/Argmax:", np.argmin(data), np.argmax(data))

# ---------------------------------
print("\n============================")
print("📘 7️⃣ TRIGONOMETRY, LOG, ROUND")
print("============================")
angles = np.array([0, np.pi/2, np.pi])
print("sin:", np.sin(angles))
print("cos:", np.cos(angles))
print("tan:", np.tan(angles))
print("exp:", np.exp(data))
print("log:", np.log(data))
print("round(π,3):", np.round(np.pi, 3))

# ---------------------------------
print("\n============================")
print("📗 8️⃣ ARRAY MANIPULATION")
print("============================")
arrA = np.array([[1, 2], [3, 4]])
arrB = np.array([[5, 6]])
print("Concatenate:\n", np.concatenate((arrA, arrB), axis=0))
print("Transpose:\n", np.transpose(arrA))
print("Flatten:", arrA.ravel())

# ---------------------------------
print("\n============================")
print("📘 9️⃣ LINEAR ALGEBRA")
print("============================")
A = np.array([[1, 2], [3, 4]])
print("Dot:\n", np.dot(A, A))
print("Inverse:\n", np.linalg.inv(A))
print("Determinant:", np.linalg.det(A))
print("Eigenvalues/Vectors:\n", np.linalg.eig(A))
print("Norm:", np.linalg.norm(A))

# ---------------------------------
print("\n============================")
print("📗 🔟 RANDOM MODULE")
print("============================")
np.random.seed(42)
print("Random (0-1):\n", np.random.rand(3,3))
print("Random Integers:\n", np.random.randint(1,10,5))
print("Normal Distribution:\n", np.random.randn(2,3))
print("Random Choice:", np.random.choice([10, 20, 30, 40]))

# ---------------------------------
print("\n============================")
print("📘 1️⃣1️⃣ BROADCASTING")
print("============================")
A = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2,3])
print("A + b =\n", A + b)
print("A * 2 =\n", A * 2)

# ---------------------------------
print("\n============================")
print("📗 1️⃣2️⃣ INPUT/OUTPUT")
print("============================")
arr_save = np.array([1,2,3,4,5])
np.save('data.npy', arr_save)
print("Loaded from npy:", np.load('data.npy'))

np.savetxt('data.txt', arr_save)
print("Loaded from txt:", np.loadtxt('data.txt'))

# ---------------------------------
print("\n============================")
print("📘 1️⃣3️⃣ UTILITY FUNCTIONS")
print("============================")
arrU = np.array([1, 2, 2, 3, np.nan, np.inf])
print("Unique:", np.unique(arrU))
print("Sort:", np.sort(arrU))
print("Where (arrU>1):", np.where(arrU > 1))
print("All True?:", np.all([True, True, False]))
print("Any True?:", np.any([False, True, False]))
print("IsNaN:", np.isnan(arrU))
print("IsInf:", np.isinf(arrU))

# ---------------------------------
print("\n============================")
print("📗 1️⃣4️⃣ PYTHON LAMBDA, MAP, FILTER, REDUCE")
print("============================")
nums = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x*x, nums))
evens = list(filter(lambda x: x % 2 == 0, nums))
total = reduce(lambda a,b: a+b, nums)
print("map (squared):", squared)
print("filter (evens):", evens)
print("reduce (sum):", total)

# Equivalent NumPy
arr = np.array(nums)
print("Vectorized square:", arr**2)
print("Vectorized filter (mask):", arr[arr%2==0])

# ---------------------------------
print("\n============================")
print("📘 1️⃣5️⃣ np.vectorize vs lambda/map")
print("============================")
py_func = lambda x: x**3 + 1
vec_func = np.vectorize(py_func)
print("np.vectorize result:", vec_func(arr))

# Compare with fast vectorized NumPy
print("Fast NumPy equivalent:", arr**3 + 1)

# ---------------------------------
print("\n============================")
print("📗 1️⃣6️⃣ PERFORMANCE COMPARISON")
print("============================")
setup = "import numpy as np; arr=np.arange(10000); lst=list(arr)"
t1 = timeit.timeit("x=[i*i for i in lst]", setup=setup, number=100)
t2 = timeit.timeit("x=list(map(lambda i:i*i,lst))", setup=setup, number=100)
t3 = timeit.timeit("x=arr*arr", setup=setup, number=100)
print(f"List comprehension: {t1:.4f}s")
print(f"Map + lambda: {t2:.4f}s")
print(f"Numpy vectorized: {t3:.4f}s")

# ---------------------------------
print("\n============================")
print("📘 1️⃣7️⃣ SIMPLE ML-LIKE EXAMPLE")
print("============================")
X = np.array([[5, 8], [6, 9], [8, 5], [9, 6]])
y = np.array([80, 85, 78, 90])
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
w = np.array([0.4, 0.6])
y_pred = np.dot(X_norm, w)
print("Predicted:", y_pred)

# ---------------------------------
print("\n============================")
print("📗 1️⃣8️⃣ MELT FUNCTION (PANDAS)")
print("============================")
data = {
    'Student': ['A', 'B', 'C'],
    'Math': [85, 90, 95],
    'Science': [80, 88, 92],
    'English': [78, 85, 89]
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

melted_df = pd.melt(df, id_vars=['Student'], var_name='Subject', value_name='Marks')
print("\nMelted DataFrame:\n", melted_df)

# ---------------------------------
print("\n============================")
print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
print("============================")
