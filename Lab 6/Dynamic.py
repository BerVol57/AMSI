from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import *
import pandas as pd
import numpy as np
import copy
import sys

from matplotlib.animation import FuncAnimation

def get_trainNtest(data):
    return train_test_split(data['x'].values, 
                            data['y'].values, test_size=0.25, shuffle=True)

def mse(model, params, X, y):
    return np.mean((model(params, X) - y)**2)

## Диференційна еволюція
def DE_dynamic(data, model, bounds, mut=0.8, crossp=0.7, popsize=100, max_iter=100):
    X_train, X_test, y_train, y_test = get_trainNtest(data)
    
    func = lambda p: mse(model, p, X_train, y_train)
    
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(max_b - min_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([func(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    best_history_position = [best]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    x_plot = np.linspace(data['x'].min(), data['x'].max(), 100)
    
    def update(frame):
        nonlocal pop, fitness, best_idx, best

        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = func(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

        best_history_position.append(best)
        
        # Очистка
        axes[0].cla()
        axes[1].cla()
        
        mse_val = mse(model, best, X_train, y_train)
        param_str = "\n".join([f"p{i+1} = {v:.4f}" for i, v in enumerate(best)])
        axes[0].text(0.05, 0.95, f"Best MSE: {mse_val:.6f}\n{param_str}", 
                     transform=axes[0].transAxes,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Графік моделі
        axes[0].scatter(X_train, y_train, label='Тренування', alpha=0.6)
        axes[0].scatter(X_test, y_test, label='Тест', alpha=0.6)
        axes[0].plot(x_plot, model(best, x_plot), 'r--', label='DE')
        axes[0].set_title("Задані точки та функція регресії")
        axes[0].legend()
        axes[0].grid()

        # Графік збіжності
        mse_values = [mse(model, p, X_train, y_train) for p in best_history_position]
        axes[1].plot(mse_values, label='DE')
        axes[1].set_title("Графік збіжності")
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Ітерації')
        axes[1].set_ylabel('MSE')
        axes[1].legend()
        axes[1].grid()

        fig.suptitle(f'Ітерація {frame+1}/{max_iter}', fontsize=14)

    anim = FuncAnimation(fig, update, frames=range(max_iter), interval=200, repeat=False)
    # anim.save('de_animation.mp4', writer='ffmpeg')
    plt.tight_layout()
    plt.show()


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
    

def PSO_dynamic(data, model, bounds, a1=1.5, a2=1.5, pop_size=100, max_iter=100):
    X_train, X_test, y_train, y_test = get_trainNtest(data)
    func = lambda p: mse(model, p, X_train, y_train)

    dim = bounds.shape[0]
    minx, maxx = bounds.T
    minv, maxv = (minx - maxx) / 10, (maxx - minx) / 10

    swarm = [Particle(func, dim, minx, maxx, minv, maxv) for _ in range(pop_size)]
    best_swarm_pos = np.zeros(dim)
    best_swarm_fitnessVal = sys.float_info.max

    best_positions = []
    fitness_history = []

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x_plot = np.linspace(data['x'].min(), data['x'].max(), 100)

    def update(frame):
        nonlocal best_swarm_pos, best_swarm_fitnessVal

        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            swarm[i].velocity = (
                swarm[i].velocity +
                a1 * r1 * (swarm[i].best_part_pos - swarm[i].position) +
                a2 * r2 * (best_swarm_pos - swarm[i].position)
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

        best_positions.append(copy.copy(best_swarm_pos))
        fitness_history.append(best_swarm_fitnessVal)

        # Малювання
        axes[0].cla()
        axes[1].cla()
        
        mse_val = mse(model, best_swarm_pos, X_train, y_train)
        param_str = "\n".join([f"p{i+1} = {v:.4f}" for i, v in enumerate(best_swarm_pos)])
        axes[0].text(0.05, 0.95, f"Best MSE: {mse_val:.6f}\n{param_str}", 
                     transform=axes[0].transAxes,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[0].scatter(X_train, y_train, label='Тренування', alpha=0.6)
        axes[0].scatter(X_test, y_test, label='Тест', alpha=0.6)
        axes[0].plot(x_plot, model(best_swarm_pos, x_plot), 'g--', label='PSO')
        axes[0].set_title("Задані точки та функція регресії")
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(fitness_history, label='PSO')
        axes[1].set_yscale('log')
        axes[1].set_title("Графік збіжності")
        axes[1].set_xlabel('Ітерації')
        axes[1].set_ylabel('MSE')
        axes[1].legend()
        axes[1].grid()

        fig.suptitle(f'Ітерація {frame + 1}/{max_iter}', fontsize=14)


    anim = FuncAnimation(fig, update, frames=range(max_iter), interval=200, repeat=False)
    # anim.save('pso_animation.mp4')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data1 = pd.read_excel('DataRegression.xlsx', sheet_name='Var01')
    data2 = pd.read_excel('DataRegression.xlsx', sheet_name='Var02')
    data3 = pd.read_excel('DataRegression.xlsx', sheet_name='Var03')
    
    DE_dynamic(data1, model1, bounds1)
    PSO_dynamic(data1, model1, bounds1)
    
    DE_dynamic(data2, model2, bounds2)
    PSO_dynamic(data2, model2, bounds2)
    
    DE_dynamic(data3, model3, bounds3)
    PSO_dynamic(data3, model3, bounds3)