import numpy as np 

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# result = []

# for i in arr :
#     if i % 2 == 0 :
#         result.append(i)
# print(result)

# res = arr[arr % 2 == 0]
# print(list(res))
# print(res.tolist())


# a = np.arange(6)
# a.reshape(2, 3)
# print(a)

# b = np.vstack((a, a))
# print(b)

# c = np.hstack((a, a))
# print(c)

# a = np.array([1,2,3])
# # b = a.view()
# b = a
# b[0] = 100
# print(a)
# print(b)

# c = a.copy()
# c[0] = 500
# print(a)
# print(c)


# a = np.random.rand(10)
# b = np.random.randint(10)
# c = np.random.randn(1,10,4)

# print(a)
# print(b)
# print(c)

# a = np.array([1,2,3,4])
# b = np.array([10,20,30,40])

# print(a+b)
# print(a*b)
# print(a**b)
# print(np.sqrt(b))



# a = np.array([[1],[2],[3]])
# b = np.array([10,20,30])

# print(a+b)


sales = np.array([[1200,1500,1000,1800],[800,950,1100,1200],[1500,1600,1700,1750]])

TOTAL_sales = sales.sum()
AVG_sales = sales.mean()
MAX_sales = sales.max(0)
STD_sales = sales.std()
MIN_sales = sales.min()

print("Total sales for each region : ",TOTAL_sales)
print("AVG sales across all region :",AVG_sales)
print("Highest sales month (col-wise MAX) : ",MAX_sales)
print("Standard deviation of sales value : ",STD_sales)
print("MIN sales value in the dataset : ",MIN_sales)