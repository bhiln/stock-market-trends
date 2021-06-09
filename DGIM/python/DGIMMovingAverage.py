#!/usr/bin/env python
# coding: utf-8

# ## DGIM Moving Average Algorithms

# In[1]:


import math
import copy

class DGIM:
    
    def __init__(self, num_buckets, bit_depth, snapshot):
        self.buckets = []
        self.times = {}
        self.bits = []
        self.value = 0
        for b in range(num_buckets):
            self.buckets.append(0)
        for d in range(bit_depth):
            self.bits.append(0)
        self.num_buckets = num_buckets
        s = copy.deepcopy(snapshot)
        s.reverse()
        for b in s:
            self.add_bit(b)
    
    def calc_value(self):
        v = 0
        for i in range(self.num_buckets):
            if i == self.num_buckets-1 and self.buckets[i] > 0:
                v += pow(2,i)*self.buckets[i]
                v -= pow(2,i)/2
            elif (self.buckets[i] > 0):
                v += pow(2,i)*self.buckets[i]
                if (self.buckets[i+1] == 0):
                    v -= pow(2,i)/2
        self.value = math.ceil(v)
        
    def add_bit(self,bit):
        
        self.bits.insert(0,bit);
        self.bits = self.bits[0:-1]
        
        if bit == 1:
            self.buckets[0] += 1
        for i in range(self.num_buckets):
            if self.buckets[i] > 2:
                self.buckets[i] -= 2
                try:
                    self.buckets[i+1] += 1
                except:
                    pass
        if self.buckets[-1] > 2:
            self.buckets[-1] = 2
        
        self.calc_value()


# In[2]:


class DGIMMovingAverage:
    
    def __init__(self, stock_binary, window_size, num_buckets, bit_depth, ground_truth=None, quiet=True):
        self.streams = []
        self.mov_avg = []
        self.error = []
        for i in range(bit_depth):
            tmp = []
            for w in range(window_size):
                tmp.append(0)
            self.streams.append(tmp)

        for d in range(len(stock_binary)):
            day = stock_binary[d]
            counts = []
            for i in range(len(day)):
                self.streams[i].insert(0,day[i])
                self.streams[i] = self.streams[i][0:window_size]
                tmp = DGIM(num_buckets, bit_depth, self.streams[i])
                counts.insert(0,tmp.value)
            total = 0
            for i in range(len(counts)):
                total += counts[i]*pow(2,i)
            if not quiet:
                print(d, ground_truth[d], total/window_size, 100*(total/window_size-ground_truth[d])/ground_truth[d])
            self.mov_avg.append(total/window_size if d >= window_size else None)
            self.error.append(100*(total/window_size-ground_truth[d])/ground_truth[d])


# In[ ]:




