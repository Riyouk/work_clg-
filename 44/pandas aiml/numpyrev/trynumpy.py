# import array as arr
# arr2 = arr.array('i',[1,2,3,4,5])
# print(arr2)
import numpy as np 
# arr1 = np.arange(10)
# arr1 = np.arange(0,10,2)
# arr1 = np.arange(0,1,0.1)
# print(arr1)
# arr2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
# print(arr2)
# arr3 = np.linspace(1,100,5)
# print(arr3)
# arr4 = np.zeros((2,3))
# arr5 = np.ones((3,3))
# arr6 = np.full((2,2),8)
# arr7 = np.eye(3)
# print(arr4)
# print(arr5)
# print(arr6)
# print(arr7)
# print(arr7.shape)
# print(arr7.ndim)
# print(arr7.size)
# arr8 = np.array([[[[1,2],[3,4]],[[5,6],[7,8]]]])
# print(arr8.ndim)

#qddition
# arr1 = np.array([1,2,3])
# arr2 = np.array([4,5,6])
# arr3 = arr1+arr2
# ans = arr3
# print(sum(arr1))
# print(ans)

# subtraction 
# multiplication
# division 

# arr1 = np.random.rand(3,3,5)
# # arr2 = np.random.randn(4,5,6)
# arr3 = np.random.randint(0,100,(4,4))
# arr4 = np.random.randint(0,100,(3,3))
# # arr5 = 
# # print(arr1)
# # print(arr2)
# print(arr3)
# print(np.sum(arr4,axis=1))

# array = np.array([5,10,15,20])
# squarearr = np.power(array,2)
# squarearr = np.power(array,3)
# subtract = np.subtract(array,10)
# # trying = subtract[subtract > 15 ]=0
# # print(trying)
# array[array > 15] = 0
# print(array)

import time 
# start_tym = time.time()
# list1 = [1,2,3,4,5,6]
# sum = 0
# for i in range(len(list1)):
#     sum = sum+i
# end_tym = time.time()
# print(sum)
# print(f"start_tym {start_tym:2f}")
# print(f"end_tym {end_tym:2f}")
# print(f"total tym taken = {start_tym-end_tym:2f}" )

# start_tym = time.time()
# arr = np.random.randint(0,1000000,1000000)
# square = np.power(arr,2)
# print(square)
# print(arr)
# end_tym = time.time()
# print(f"start_tym {start_tym:2f}")
# print(f"end_tym {end_tym:2f}")
# print(f"total tym taken = {start_tym-end_tym:2f}" )

# start_tym = time.time()
# list1 = []
# sum = 0
# for i in range(len(arr)):
#     list1.append(arr**2)    
# end_tym = time.time()
# print(f"start_tym {start_tym:2f}")
# print(f"end_tym {end_tym:2f}")
# print(f"total tym taken = {start_tym-end_tym:2f}" )

# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
# total_col = np.sum(arr,axis=0)
# total_row = np.sum(arr,axis=1)
# print(total_col)
# forex = arr.ravel()
# print(np.split(forex,2))  
# print(arr.ravel()) # 1d view 
# print(arr.flatten()) #1d with copy
# print(forex.reshape(3,3))

# print(total_row)
# print(arr)

#concatinate
# arr1 = np.array([[1,2],[3,4]])
# arr2 = np.array([[5,6],[7,8]])
# # arr2= np.array([[5,6]])
# print(arr.ndim)
# result = np.concatenate((arr1,arr2),axis=1)
# print(result)

#map,filter,reduce,lamda

# arr = np.random.randint(100,200,50)
# # # print(arr)
# result = list(filter(lambda x : x > 150 ,arr))
# # result1= list(map(lambda x : x > 150 ,arr))
# # print(result1)
# # print(result)

# from functools import reduce
# no_elements = reduce(lambda x ,_: x + 1 ,result,0)
# print(no_elements)

# import math 
# arr = np.random.randint(100,500,100)
# squares_root = list(filter(lambda x : math.isqrt(x)**2 == x ,arr))
# print(squares_root)
# no_elements = reduce(lambda  x,_: x + 1 ,squares_root,0)
# print(no_elements)




#Creating a 2D array consisting car names, horsepower and acceleration
car_names = ['chevrolet chevelle malibu', 'buick skylark 320', 'plymouth satellite', 'amc rebel sst', 'ford torino']
horsepower = [130, 165, 150, 150, 140]
acceleration = [18, 15, 18, 16, 17]
car_hp_acc_arr = np.array([car_names, horsepower, acceleration])
#Accessing name and horsepower 
print(car_hp_acc_arr)
print(car_hp_acc_arr[0:2,0:2])
print(car_hp_acc_arr[0:3,0:3])