#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Model
from smt_training import train_smt
from backpropagation_training import train_backpropagation
from testing import test


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

points = []
for i in range(0, 360, 90):
    for j in range(0, 180, 60):
        i_radians = (i * np.pi) / 180.0
        j_radians = (j * np.pi) / 180.0
        x = np.cos(i_radians) *  np.sin(j_radians)
        y = np.sin(i_radians) * np.sin(j_radians)
        z = np.cos(j_radians)
        points.append([x, y, z])
points = torch.FloatTensor(points)
print("points.shape: ", points.shape)

hyperplane1 = [random.uniform(0, 0.001), random.uniform(0, 0.001), random.uniform(0, 0.001)]
print("hyperplane1: ", hyperplane1)

hyperplane2 = [random.uniform(0, 0.001), random.uniform(0, 0.001), random.uniform(0, 0.001)]
print("hyperplane1: ", hyperplane2)

labels = []
for point in points:
    if (point[0] * hyperplane1[0] + point[1] * hyperplane1[1] + point[2] * hyperplane1[2] > 0) and (point[0] * hyperplane2[0] + point[1] * hyperplane2[1] + point[2] * hyperplane2[2] > 0):   
        labels.append(1)
    else:
        labels.append(0)

labels = torch.tensor(labels)

criterion = nn.CrossEntropyLoss()
num_epochs = 100


# In[3]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1, 1)

plane1_X, plane1_Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
plane1_Z = (-hyperplane1[0] * plane1_X - hyperplane1[1] * plane1_Y) / hyperplane1[2]

ax.plot_surface(plane1_X, plane1_Y, plane1_Z, color='purple', alpha=0.5)

plane2_X, plane2_Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
plane2_Z = (-hyperplane2[0] * plane2_X - hyperplane2[1] * plane2_Y) / hyperplane2[2]

ax.plot_surface(plane2_X, plane2_Y, plane2_Z, color='orange', alpha=0.5)

U, V = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
sphere_X = np.cos(U) * np.sin(V)
sphere_Y = np.sin(U) * np.sin(V)
sphere_Z = np.cos(V)

ax.plot_surface(sphere_X, sphere_Y, sphere_Z, color='green', alpha=0.3)

for (index, point) in enumerate(points):
    if labels[index] == 1:
        ax.scatter(point[0], point[1], point[2], color='blue', alpha = 1.0)
    else:
        ax.scatter(point[0], point[1], point[2], color='red', alpha = 1.0)

plt.show()


# In[4]:


embedding_sizes = [10, 20, 30, 40, 50, 60, 70, 80]
smt_times = []
backpropagation_times = []

for embedding_size in embedding_sizes:
    
    print("embedding_size: ", embedding_size)

    model = Model(embedding_size)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    model.to(device)

    start_time = time.time()
    train_backpropagation(model, points, labels, criterion, optimizer, num_epochs)
    backpropagation_times.append(time.time() - start_time)

    start_time = time.time()
    train_smt(model, points, labels)
    smt_times.append(time.time() - start_time)
    test(model, points, labels, criterion)


# In[5]:


for i in range(len(embedding_sizes)):
    embedding_sizes[i] = 3 * embedding_sizes[i] + 2 * embedding_sizes[i]


# In[6]:


plt.scatter(embedding_sizes, smt_times)
plt.title("SMT Training Time")
plt.xlabel("Number of Model Weights")
plt.ylabel("Running Time (secs)")
plt.show()


# In[7]:


plt.scatter(embedding_sizes, backpropagation_times)
plt.title("Backpropagation Training Time")
plt.xlabel("Number of Model Weights")
plt.ylabel("Running Time (secs)")
plt.show()


# In[8]:


embedding_sizes = [100000, 1000000, 10000000, 100000000]
smt_times = []
backpropagation_times = []

for embedding_size in embedding_sizes:
    
    print("embedding_size: ", embedding_size)

    model = Model(embedding_size)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    model.to(device)

    start_time = time.time()
    train_backpropagation(model, points, labels, criterion, optimizer, num_epochs)
    backpropagation_times.append(time.time() - start_time)


# In[9]:


for i in range(len(embedding_sizes)):
    embedding_sizes[i] = np.log10(3 * embedding_sizes[i] + 2 * embedding_sizes[i])


# In[10]:


plt.scatter(embedding_sizes, backpropagation_times)
plt.title("Backpropagation Training Time")
plt.xlabel("Log10 Number of Model Weights")
plt.ylabel("Running Time (secs)")
plt.show()


# In[ ]:




