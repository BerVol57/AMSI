import numpy as np
import copy
import sys

def max_with_key(a, f):
    mapped_values = np.array([f(x) for x in a])
    max_index = np.argmax(mapped_values)
    return max_index

def min_with_key(a, f):
    mapped_values = np.array([f(x) for x in a])
    min_index = np.argmin(mapped_values)
    return min_index

### 
def Cuckoo(f, max_iter, n, dim, minx, maxx, 
           p_detect, delta=.5):
    minx, maxx = np.array(minx), np.array(maxx)
    SWRM = []
    BP = []
    BCV = []
    
    cuckoos = np.random.uniform(minx, maxx, (n, dim))
    
    best_cuckoo = cuckoos[min_with_key(cuckoos, f)]
    best_value = f(best_cuckoo)
    
    
    for iteration in range(max_iter):
        # for _ in range(n):
            
        k = np.random.randint(0, n)
        
        x_cur = cuckoos[k] + np.random.uniform(-delta, delta)*(maxx-minx)
        x_cur = np.clip(x_cur, minx, maxx)
        
        f_x_cur = f(x_cur)
        if f_x_cur < best_value:
            best_value = copy.copy(f_x_cur)
            best_cuckoo = copy.copy(x_cur)
            cuckoos[k] = x_cur
            

        if np.random.rand() < p_detect:
            m = max_with_key(cuckoos, f)
            cuckoos[m] += (minx - maxx) * np.random.uniform(-delta, delta)
            cuckoos[m] = np.clip(cuckoos[m], minx, maxx)
            
            # fcm = f(cuckoos[m])
            # if fcm < best_value:
            #     best_value = fcm
            #     best_cuckoo = cuckoos[m]
            
        SWRM.append(copy.copy(cuckoos))
        BP.append(copy.copy(best_cuckoo))
        BCV.append(copy.copy(best_value))
        
    return BCV, BP, SWRM

### 
def Bat(f, max_iter, n, dim, minx, maxx, 
        loudness=.5, pulse_rate=.5):
    minx, maxx = np.array(minx), np.array(maxx)
        
    SWRM = []
    BP = []
    BBV = []
    
    v = np.zeros((n, dim))
    
    bats = np.random.uniform(minx, maxx, (n, dim))
    
    best_bat = bats[min_with_key(bats, f)]
    best_value = f(best_bat)
    
    for iteration in range(max_iter):
        current_loudness = loudness*(1-np.exp(-pulse_rate*iteration))
        
        for i in range(n):
            frequency = .5
            v[i] += (bats[i] - best_bat) * frequency
            v[i] = np.clip(v[i], minx/2, maxx/2)
            
            bats[i] += v[i]
            bats[i] = np.clip(bats[i], minx, maxx)
            
            if np.random.rand() > current_loudness:
                bats[i] = best_bat + .001 * np.random.rand(dim)
            
            fbi = f(bats[i])
            if fbi < best_value:
                best_value = copy.copy(fbi)
                best_bat = copy.copy(bats[i])
                
                
        
        SWRM.append(copy.copy(bats))
        BP.append(copy.copy(best_bat))
        BBV.append(copy.copy(best_value))
    
    return BBV, BP, SWRM