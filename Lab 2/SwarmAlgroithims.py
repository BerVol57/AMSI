import numpy as np
import copy
import sys


### PSO
class Particle:
    def __init__(self, fitness, dim, minx, maxx, minv, maxv):
        
        self.position = np.random.uniform(minx, maxx)
            
        self.velocity = np.random.uniform(minv, maxv)
        
        while fitness(self.position) == sys.float_info.max:
            self.position = np.random.uniform(minx, maxx)
        
        self.fitness = fitness(self.position)
        
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness
    


def PSO(fitness, max_iter, n, dim, minx, maxx, minv, maxv, a1 = 1.49445, a2 = 1.49445, viz=False):
    SWRM = []
    BP = []
    BSF = []
    # create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, minv, maxv) for _ in range(n)]    
    # compute best_pos & best_fit
    best_swarm_pos = np.zeros(dim)
    
    best_swarm_fitnessVal = sys.float_info.max
    
    for iteration in range(max_iter):
        for i in range(n):
            

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

        
            swarm[i].velocity = (
                (swarm[i].velocity) + 
                (a1 * r1 * (swarm[i].best_part_pos - swarm[i].position)) + 
                (a2 * r2 * (best_swarm_pos - swarm[i].position))
            )
            
            swarm[i].velocity = np.clip(swarm[i].velocity, minv, maxv)
                
            swarm[i].position += swarm[i].velocity
            
            for k in range(dim):
                if swarm[i].position[k] < minx[k]:
                    swarm[i].position[k] = minx[k] + abs(swarm[i].position[k] - minx[k])
                    swarm[i].velocity[k] *= -1
                elif swarm[i].position[k] > maxx[k]:
                    swarm[i].position[k] = maxx[k] + abs(swarm[i].position[k] - maxx[k])
                    swarm[i].velocity[k] *= -1
            
            swarm[i].fitness = fitness(swarm[i].position)
            
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)
            
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)
        SWRM.append(np.array([p.position for p in swarm]))
        BP.append(best_swarm_pos)
        BSF.append(copy.copy(best_swarm_fitnessVal))
    
    if viz:
        return BSF, BP, SWRM
    else:
        return BSF


### Bee
def BA(f, max_iter, n, dim, minx, maxx, L_s, L_es, z_e, z_0, delta=.1, teta_max=.9, alpha=.9, viz=False):
    SWRM = []
    BP = []
    BBF = []
    
    bees = np.random.uniform(minx, maxx, (n, dim))
    
    for i in range(n):
        while f(bees[i]) == sys.float_info.max:
            bees[i] = np.random.uniform(minx, maxx)
    
    best_pos = None
    best_fitness = sys.float_info.max
    
    for iteration in range(max_iter):
        
        fitness = np.array([f(bee) for bee in bees])
        sorted_indices = np.argsort(fitness)
        bees = bees[sorted_indices]
        
        if fitness[sorted_indices[0]] < best_fitness:
            best_fitness = fitness[sorted_indices[0]]
            best_pos = bees[0].copy()
        
        for i in range(L_es):
            for _ in range(z_e):
                teta = teta_max*(alpha**iteration)
                candidate = bees[i] + teta*(np.array(maxx) - np.array(minx))*np.random.uniform(-1, 1, dim)
                candidate = np.clip(candidate, minx, maxx)
                candidate_val = f(candidate)
                
                if candidate_val < fitness[i]:
                    bees[i] = candidate
                    fitness[i] = candidate_val
        
        for i in range(L_es, n):
            bees[i] = np.random.uniform(minx, maxx)
        
        
        SWRM.append(bees)
        BP.append(best_pos)
        BBF.append(best_fitness)
        
    if viz:
        return BBF, BP, SWRM
    else:
        return BBF


### Firefly
def FA(f, max_iter, n, dim, minx, maxx, alpha=.2, beta_max=.9, gamma=.9, viz=False):
    fireflies = np.random.uniform(minx, maxx, (n, dim))
    
    for i in range(n):
        while f(fireflies[i]) == sys.float_info.max:
            fireflies[i] = np.random.uniform(minx, maxx)
    
    best_pos = None
    best_val = sys.float_info.max
    
    SWRM = []
    BP = []
    BFA = []
    
    for iteration in range(max_iter):
        intensity = np.array([f(ff) for ff in fireflies])
        sorted_indices = np.argsort(intensity)
        fireflies = fireflies[sorted_indices]
        intensity = intensity[sorted_indices]
        
        if intensity[0] < best_val:
            best_val = intensity[0]
            best_pos = fireflies[0].copy()
            
        for i in range(n):
            for j in range(n):
                if intensity[j] < intensity[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta_max * np.exp(-gamma * r)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) \
                        + alpha * (np.random.uniform(-.5, .5, dim))
                    
                    fireflies[i] = np.clip(fireflies[i], minx, maxx)

        SWRM.append(fireflies)
        BP.append(best_pos)
        BFA.append(best_val)
    
    if viz:
        return BFA, BP, SWRM
    else:
        return BFA