import numpy as np
import copy
import sys


### PSO
class Particle:
    def __init__(self, fitness, dim, minx, maxx):
            
        self.position = np.random.uniform(minx, maxx)
            
        self.velocity = np.random.uniform(minx, maxx)
        
        self.fitness = fitness(self.position)
        
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness
    


def PSO(fitness, max_iter, n, dim, minx, maxx, a1 = 1.49445, a2 = 1.49445):
    BSF = []
    # create n random particles
    swarm = [Particle(fitness, dim, minx, maxx) for i in range(n)]
    
    # compute best_pos & best_fit
    best_swarm_pos = np.zeros(dim)
    
    best_swarm_fitnessVal = sys.float_info.max
    
    for iteration in range(max_iter):
        for i in range(n):
            

            r1 = np.random.rand()
            r2 = np.random.rand()

        
            swarm[i].velocity = (
                (swarm[i].velocity) + 
                (a1 * r1 * (swarm[i].best_part_pos - swarm[i].position)) + 
                (a2 * r2 * (best_swarm_pos - swarm[i].position))
            )
            
            swarm[i].velocity = np.clip(swarm[i].velocity, [x/2 for x in minx], [x/2 for x in maxx])
                
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
        
        BSF.append(copy.copy(best_swarm_fitnessVal))
        
    return BSF


### Bee
def BA(f, max_iter, n, dim, minx, maxx, elite_sites, elite_size):
    BBF = []
    
    bees = np.random.uniform(minx, maxx, (n, dim))
    best_pos = None
    best_fitness = sys.float_info.max
    
    for iteration in range(max_iter):
        
        fitness = np.array([f(bee) for bee in bees])
        sorted_indices = np.argsort(fitness)
        bees = bees[sorted_indices]
        
        if fitness[sorted_indices[0]] < best_fitness:
            best_fitness = fitness[sorted_indices[0]]
            best_pos = bees[0].copy()
        
        for i in range(elite_sites):
            for _ in range(elite_size):
                candidate = bees[i] + np.random.uniform(-.1, .1, dim)
                candidate = np.clip(candidate, minx, maxx)
                candidate_val = f(candidate)
                
                if candidate_val < fitness[i]:
                    bees[i] = candidate
                    fitness[i] = candidate_val
        
        for i in range(elite_sites, n):
            bees[i] = np.random.uniform(minx, maxx)

        
        BBF.append(best_fitness)
    
    return BBF


### Firefly
def FA(f, max_iter, n, dim, minx, maxx, alpha=.2, beta0=1., gamma=1.):
    fireflies = np.random.uniform(minx, maxx, (n, dim))
    
    best_pos = None
    best_val = sys.float_info.max
    
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
                    beta = beta0 * np.exp(-gamma * r**2)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) \
                        + alpha * (np.random.rand(dim) - .5)
                    
                    fireflies[i] = np.clip(fireflies[i], minx, maxx)
        
        BFA.append(best_val)
    
    return BFA