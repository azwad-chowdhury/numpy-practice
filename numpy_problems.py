# source : https://www.machinelearningplus.com/python/101-numpy-exercises-python/

# Numpy practice problems

import numpy as np # importing numpy 

# problem : 02
# Create 1D array

# a = np.arange(10)
# print(a)

# problem : 03
# Create a boolean array

# a = np.ones((3,3),dtype = bool) #defining (row,column) and data type
# print(a)

# problem : 04
# Extract all odd numbers from arr

# a = np.arange(10)
# ans = a[a%2 == 1]
# print(ans)

# problem : 05
# Replace all odd numbers in arr with -1

# a = np.arange(10)
# a[a%2 == 1] = -1
# print(a)

# problem : 06
# Replace all odd numbers in arr with -1 without changing arr

# a = np.arange(10)
# out = np.where(a%2==1,-1,a)
# print(a)
# print(out)

# problem : 07
# Convert a 1D array to a 2D array with 2 rows

# a = np.arange(10)
# a = a.reshape(2,-1)
# print(a)

# problem : 08
# Stack arrays a and b vertically

# a = np.arange(1,10)
# b = np.arange(10,19)

# out = np.vstack([a,b])
# print(out)

# problem : 09
# Stack the arrays a and b horizontally


# a = np.arange(1,10,2)
# b = np.arange(0,9,1)

# out = np.hstack([a,b])
# print(out)

# problem : 10
# Create the following pattern without hardcoding. Use only numpy functions and the below input array a.
# Input:
# a = np.array([1,2,3])

# a = np.array([1,2,3])
# a = np.r_[np.repeat(a,3),np.tile(a,3)]
# print(a)


# problem : 11
# Get the common items between a and b

# a = np.arange(1,10)
# b = np.arange(8,17)
# out = np.intersect1d(a,b)
# print(out)

# Problem : 12
# From array a remove all items present in array b

# a = np.arange(2,10)
# b = np.arange(1,9)

# out = np.setdiff1d(a,b)

# print(out)
# print(out.dtype)

# Problem : 13
# Get the positions where elements of a and b match

# a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])

# out = np.where(a == b)
# print(a)
# print(b)
# print(out)

# Problem : 14
# Get all items between 5 and 10 from a

# a = np.array([2,1,3,4,8,6,90,3,7])

# out = a[(a>=5)&(a<10)]
# print(sorted(out))

# Problem : 15
# Convert the function maxx that works on two scalars, to work on two arrays.

# def maxx(x,y):

# 	if(x>=y):
# 		return x
# 	else:
# 		return y 

# pair_max = np.vectorize(maxx,otypes = [int])

# a = np.array([1,4,2,9,8,5])
# b = np.array([4,2,1,9,4,2])

# print(a)
# print(b)
# print('max of above two vector is: \t')
# print(pair_max(a,b))


# Problem : 16
# Swap columns 1 and 2 in the array arr.

# a = np.arange(9).reshape(3,-1)
# print('Original one: ',a)
# swaped_col = a[:,[1,0,2]]
# print('After swaping column: ',swaped_col)

# Problem : 17
# Swap rows 1 and 2 in the array arr:

# a = np.arange(9).reshape(3,-1)
# print('Original One: ',a)
# swaped_row = a[[1,0,2],:]
# print('Afrer swaping rows: ',swaped_row)

# Problem : 18
# Reverse the rows of a 2D array arr

# a = np.arange(9).reshape(3,-1)
# print('Original one : \t',a)
# reversed_row = a[::-1]
# print('After reversing row: ',reversed_row)


#  Problem : 19
# Reverse the columns of a 2D array arr

# a = np.arange(9).reshape(3,-1)
# print('Original one: ',a)
# reversed_col = a[:,::-1]
# print('After reversing columns: ',reversed_col)

# Problem : 20
# Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.

# a = np.random.randint(5,10,(5,3))+np.random.random((5,3))
# print(a)


# Problem : 21
# Print or show only 3 decimal places of the numpy array rand_arr.
# rand_arr = np.random.random((5,3))
# np.set_printoptions(precision = 3)
# print(rand_arr)

# Problem : 22
# Print rand_arr by suppressing the scientific notation (like 1e10)
# np.random.seed(100)
# rand_arr = np.random.random([3,3])/1e3
# print(rand_arr)


# Problem : 23
# Limit the number of items printed in python numpy array a to a maximum of 6 elements.
# a = np.arange(15)
# np.set_printoptions(threshold=6)
# print(a)


# Problem : 24
# Print the full numpy array a without truncating.
# np.set_printoptions(threshold = 6)
# a = np.arange(15)
# np.set_printoptions(threshold = np.nan)
# print(a)


# Problem : 25
# How to import a dataset with numbers and texts keeping the text intact in python numpy?
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url,delimiter = ',', dtype = 'object')
# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# print(iris[:3]) #Printing the first three row of Iris dataset


# # Problem : 26
#  Extract the text column species from the 1D iris imported in previous question
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_1d = np.genfromtxt(url,delimiter = ',',dtype = None)
# print(iris_1d.shape)
# species = np.array([row[4] for row in iris_1d])
# print(species[:4])


# Problem : 27
# Convert the 1D iris to 2D array iris_2d by omitting the species text field.
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_1d = np.genfromtxt(url,delimiter = ',',dtype = None)
# print(iris_1d)
# iris_2d = np.array([row.tolist()[:4] for row in iris_1d])
# print(iris_2d)

# Problem : 28
# Find the mean, median, standard deviation of iris's sepallength (1st column)
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_1d = np.genfromtxt(url,delimiter = ',',dtype = object)
# sepallength = np.genfromtxt(url,delimiter=',',dtype= float,usecols = [0])
# mean,med,std = np.mean(sepallength),np.median(sepallength),np.std(sepallength)

# print(mean,med,std)

# Problem : 29
# Create a normalized form of iris's sepallength whose values range exactly between 0 and 1 so that the minimum has value 0 and maximum has value 1.
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# sepallength = np.genfromtxt(url,delimiter = ',',dtype= 'float',usecols = [0])
# print(sepallength)
# Smax,Smin = sepallength.max(),sepallength.min()
# S = (sepallength-Smin)/(Smax - Smin)
# S_alt = (sepallength - Smin)/sepallength.ptp()
# print(S)
# print(S_alt)



# Problem : 30
# Compute the softmax score of sepallength.
# url  		=   'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris 		= 	np.genfromtxt(url,delimiter=',',dtype='object')
# # sepallength = 	np.array([float(row(0))] for row in iris)
# # print(iris)
# sepallength = np.genfromtxt(url,delimiter=',',dtype = float,usecols=[0])
# print(sepallength[:3])


# def softmax(x):
# 	e_x = np.exp(x - np.max(x))
# 	return e_x/e_x.sum(axis = 0 )

# print(softmax(sepallength))



# Problem : 31
# Find the 5th and 95th percentile of iris's sepallength
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url,delimiter=',',dtype='object')
# # sepallength = np.array([float(row(0))] for row in iris)

# sepallength = np.genfromtxt(url,delimiter = ',', dtype = 'float', usecols = [0])
# print(iris)
# print('\t')
# print(sepallength)

# percentile_sepal  = np.percentile(sepallength,q = [5,95])
# print("5th and 95th percentile of iris's sepallength are: ",percentile_sepal)


# Problem : 32
# Insert np.nan values at 20 random positions in iris_2d dataset
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
# print(iris_2d)

# # Method 01
# i,j = np.where(iris_2d)
# # i, j contain the row numbers and column numbers of 600 elements of iris_x
# np.random.seed(100)
# iris_2d[np.random.choice((i),20),np.random.choice((j),20)] = np.nan
# print(iris_2d[:10])

# Method 02
# np.random.seed(100)
# iris_2d[np.random.randint(150,size = 20),np.random.randint(4,size = 20)] = np.nan
# print(iris_2d[:10])


# Problem : 33
# Find the number and position of missing values in iris_2d's sepallength (1st column)
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_2d= np.genfromtxt(url,delimiter=',',dtype = 'float',usecols=[0,1,2,3])
# # sepallength = np.genfromtxt(url,delimiter = ',', dtype = 'float', usecols = [0])
# iris_2d[np.random.randint(150,size= 20),np.random.randint(4,size = 20)] = np.nan
# print(iris_2d[:,0])

# print('number of the missing values are: ',np.isnan(iris_2d[:, 0]).sum())
# print('position of the missing values are: ',np.where(np.isnan(iris_2d[:,0])))

# problem : 34
# Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_2d= np.genfromtxt(url,delimiter=',',dtype = 'float',usecols=[0,1,2,3])

# condition = (iris_2d[:,2] > 1.5) & (iris_2d[:,0] < 5)
# printe(iris_2d[condition])


# problem : 35
# Select the rows of iris_2d that does not have any nan value
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_2d= np.genfromtxt(url,delimiter=',',dtype = 'float',usecols=[0,1,2,3])

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
# iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

# # Method 01

# any_nan_in_row = np.array([~np.any(np.isnan(row)) for row in iris_2d])
# print(iris_2d[any_nan_in_row][:5])

# Problem : 36
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_2d= np.genfromtxt(url,delimiter=',',dtype = 'float',usecols=[0,1,2,3])

# cor = np.corrcoef(iris_2d[:,0],iris_2d[:,2])[0,1]
# print(cor)

# Problem : 37
# Find out if iris_2d has any missing values

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])


# print(np.isnan(iris_2d).any())

# problem 38
# Replace all occurances of nan with 0 in numpy array

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
# iris_2d[np.random.randint(150,size = 20),np.random.randint(4,size = 20)]= np.nan

# iris_2d[np.isnan(iris_2d)] = 0
# print(iris_2d[:4])

# problem : 39
# Find the unique values and the count of unique values in iris's species

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url,delimiter = ',',dtype= 'object')
# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')


# species = np.array([row.tolist()[4] for row in iris])
# # print(species)

# unique_count = np.unique(species,return_counts = True)
# print(unique_count)


# Problem : 40
# How to convert a numeric to a categorical (text) array?

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')
# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# # Bin Petal Length
# bin_petal_length = np.digitize(iris[:,2].astype(float),[0,3,5,10])
# # print(bin_petal_length)

# # Map category name
# label_map = {1:'small',2:'medium',3:'large'}
# petal_cat = [label_map[x] for x in bin_petal_length]

# print(petal_cat)


# Problem : 41
# Create a new column for volume in iris_2d, where volume is (pi x petallength x sepal_length^2)/3

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')
# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
# # volume = (pi x petallength x sepal_length^2)/3

# petallength = iris[:,2].astype(float)
# sepallength = iris[:,0].astype(float)
# # Volume computation
# volume = (np.pi*petallength*sepallength**2)/3

# # Introduce new dimension to macth iris
# volume = volume[:,np.newaxis]

# # add the new column
# out = np.hstack([iris,volume])
# print(out)

# Problem : 42	
# Randomly sample iris's species such that setose is twice the number of versicolor and virginica

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')

# species = iris[:,4]
# # print(species)

# # Approach 1: Probabilistically
# np.random.seed(100)
# a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
# species_out = np.random.choice(a,150,p = [0.5,0.25,0.25])

# print(species_out)


# Problem : 43
# What is the value of second longest petallength of species setosa
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')
# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# print(iris[:,4])


# Get the species and petal length column
# petal_len_setosa = iris[(iris[:,4] == b'Iris-setosa'),[2]].astype('float')
# # print(petal_len_setosa)

# second_largest = np.unique(np.sort(petal_len_setosa))[-2]
# print(second_largest)

# Problem : 44
# Sort the iris dataset based on sepallength column.
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')
# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# # sepal_col = iris[:,0].astype('float')
# # # print(sepal_col)

# # sorted_sepal = np.sort(sepal_col)
# # print(sorted_sepal)

# # Sort by column via argsort
# print(iris[iris[:,0].argsort()])

# Problem : 45
# Find the most frequent value of petal length (3rd column) in iris dataset.
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')
# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# # vals,count = np.unique(iris[:,2],return_counts = True)
# # print(vals[np.argmax(count)])

# # print(np.unique(iris[:,2],return_counts = True))

# vals,count = np.unique(iris[:,2],return_counts = True)

# print(vals[np.argmax(count)])
# # print(iris[:,2])


# Problem : 46
# Find the position of the first occurrence of a value greater than 1.0 in petalwidth 4th column of iris dataset.

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')
# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

# print(np.argwhere(iris[:,3].astype('float')>1)[0])


# Problem : 47
# From the array a, replace all values greater than 30 to 30 and less than 10 to 10.

# np.set_printoptions(precision = 2)
# np.random.seed(100)
# a = np.random.uniform(1,50,20)
# # print(a)
# print(np.clip(a,a_min = 10,a_max = 30))


# Problem : 48
# Get the positions of top 5 maximum values in a given array a.
# np.random.seed(100)
# a = np.random.uniform(100,50,20)
# np.set_printoptions(precision = 2)
# print(a)

# # top_five =a[a.argsort()][-5:]
# top_five =a[a.argsort()][-5:]
# print(top_five)



# **********************************

# Problem : 49
# Compute the counts of unique values row-wise.

# np.random.seed(100)
# arr = np.random.randint(1,11,size=(6, 10))
# # print(arr)


# def count_of_all_values_rowwise(arr2d):
# 	# Unique values and counts row wise
# 	num_counts_array = [np.unique(row,return_counts = True) for row in arr2d]

# Incomplete

# **********************************

# Problem : 50
# Convert an array of arrays into a flat 1d array
# arr1 = np.arange(3)
# arr2 = np.arange(3,7)
# arr3 = np.arange(7,10)



# arr_of_arr = np.array([arr1,arr2,arr3])
# print(arr_of_arr)


# # By using np concatenation
# arr_1d = np.concatenate(arr_of_arr)
# print(arr_1d)


# ****************************************************
# Problem : 51
# Compute the one-hot encodings (dummy binary variables for each unique value in the array)
# ****************************************************
# Incomplete










































































