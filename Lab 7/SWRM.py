import numpy as np
import copy
import sys


## Диференційна еволюція
def de(func, bounds, mut=.8, crossp=.9, 
       popsize=100, max_iter=1_00):
    dimensions = len(bounds)
    # Ініціалізація популяції
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(max_b - min_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([func(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    best_history_value = []
    best_history_position = []
    
    for i in range(max_iter):
        for j in range(popsize):
            # Вибір 3 різних векторів, не включаючи j
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            
            # Мутація
            mutant = np.clip(a + mut * (b - c), 0, 1)
            
            # Кросовер
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            
            # Декодування
            trial_denorm = min_b + trial * diff
            f = func(trial_denorm)
            
            # Відбір
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        best_history_value.append(fitness[best_idx])
        best_history_position.append(best)
    return best_history_position, best_history_value 


## Рій частинок
class Particle:
    def __init__(self, fitness, dim, minx, maxx, minv, maxv):
        
        self.position = np.random.uniform(minx, maxx)
            
        self.velocity = np.random.uniform(minv, maxv)
        
        while fitness(self.position) == sys.float_info.max:
            self.position = np.random.uniform(minx, maxx)
        
        self.fitness = fitness(self.position)
        
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness
    

## func, bounds, mut=0.8, crossp=0.7, popsize=10, max_iter=100
def pso(func, bounds, a1 = 1.5, a2 = 1.5, 
        pop_size=100, max_iter=1_00):
    BP = []
    BSF = []
    # particles
    dim = bounds.shape[0]
    minx, maxx = bounds.T
    minv, maxv = (minx-maxx)/1e2, (maxx-minx)/1e2
    
    swarm = [Particle(func, dim, minx, maxx, minv, maxv) for _ in range(pop_size)]    
    # compute best_pos & best_fit
    best_swarm_pos = np.zeros(dim)
    
    best_swarm_fitnessVal = sys.float_info.max
    
    for iteration in range(max_iter):
        for i in range(pop_size):
            

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
            
            swarm[i].fitness = func(swarm[i].position)
            
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)
            
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)
        BP.append(best_swarm_pos)
        BSF.append(copy.copy(best_swarm_fitnessVal))
    
    else:
        return BP, BSF