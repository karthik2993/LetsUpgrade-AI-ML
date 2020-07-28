#!/usr/bin/env python
# coding: utf-8

# # 1. Create a 3x3x3 array with random values

# In[3]:


import numpy as np
a = np.random.random((3,3,3))
print(a)


# # # 2.Create a 5x5 matrix with values 1,2,3,4 just below the diagonal

# In[4]:


b = np.diag(1+np.arange(4), k = -1)
print (b)


# # 3.Create a 8x8 matrix and fill it with a checkerboard pattern

# In[5]:


c = np.zeros ((8,8), dtype=int)
c[1::2, ::2]= 1
c[::2, 1::2] = 1
print (c)


# # 4. Normalize a 5x5 random matrix

# In[6]:


Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z= (Z-Zmin)/(Zmax-Zmin)
print (Z)


# # 5.  How to find common values between two arrays?

# In[8]:


import numpy as np
array1 = np.array([10, 20, 40, 60,80,45])
print("Array1: ",array1)
array2 = [10, 30, 40,80,90,50,45]
print("Array2: ",array2)
print("Common values between two arrays:")
print(np.intersect1d(array1, array2))


# # 6.How to get the dates of yesterday, today and tomorrow?

# In[9]:


import numpy as np
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
print("Yestraday: ",yesterday)
today     = np.datetime64('today', 'D')
print("Today: ",today)
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print("Tomorrow: ",tomorrow)


# # 7. Consider two random array A and B, check if they are equal

# In[22]:


A1 = np.random.randint(0,2,10)
A2 = np.random.randint(0,2,10)
equal = np.allclose(A1,A2)
print(A1)
print(A2)
print(equal)


# # 8.Create random vector of size 10 and replace the maximum value by 0 

# In[29]:


vector = np.random.random(10)
print("The Random Vector :\n", vector)
vector[vector.argmax()] = 0
print("The Replace vector :\n",vector)


# # 9. How to print all the values of an array?

# In[43]:


import numpy as np 

with np.printoptions(precision = 3): 
     print(np.zeros((10,10)))


# # 10.Subtract the mean of each row of a matrix

# In[75]:


X = np.random.rand(3,3)

Y = X - X.mean(axis=1).reshape(-1, 1)

Y


# # 11.Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? 

# In[60]:


Z = np.ones(5)
print(Z)
I = np.random.randint(0,len(Z),10)
print(I)
Z += np.bincount(I, minlength=len(Z))
print(Z)


# # 12.How to get the diagonal of a dot product?

# In[79]:


A = np.random.randint(0,10,(3,3))
B= np.random.randint(0,10,(3,3))
#Slow version
print(A)
print(B)

np.diag(np.dot(A, B))
np.sum(A * B.T, axis=1)
np.einsum("ij,ji->i", A, B)


# # 13.How to find the most frequent value in an array?

# In[69]:


arr = np.random.randint(0,10,20)
print (arr)
print('rank:', np.bincount(arr).argmax())


# # 14.How to get the n largest values of an array

# In[81]:


Z = np.arange(100)
np.random.shuffle(Z)
n = 5

print (Z[np.argsort(Z)[-n:]])


# # 15.How to create a record array from a regular array?

# In[87]:


a = np.array([("Lets", 1, 3),
              ("Upgrade", 3, 2)])
b = np.core.records.fromarrays(a.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')

print(Z)
print(R)

