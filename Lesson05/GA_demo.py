import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# 参数定义
alpha = math.radians(1.5) # 坡度/°
theta = math.radians(120) # 换能器开角/°
width = 4 # 东西宽度/海里
length = 5
height_max = 197.20
height_min = 20.00
height_avg = 62.47
d_min = 2 * height_avg * math.tan(theta/2) * (1-0.5)
d_max = 2 * height_avg * math.tan(theta/2) * (1-0.4)
# d_min = 2 * height_avg * math.tan(theta/2) * (1-0.2)
# d_max = 2 * height_avg * math.tan(theta/2) * (1-0.1)
pi = math.pi

# 导入数据
df = pd.read_excel('data.xlsx')

# 提取 x 坐标点, y 坐标点, 海水深度数据
x = df.columns[1:]
y = df.index
elevation = df.iloc[:, 1:].values

print(len(x),len(y), len(elevation[0]), len(elevation))
# print(x)
# print(y.values)
# print(elevation)

# 遗传算法参数
population_size = 100
num_generations = 20
elite_size = 10
mutation_rate = 0.1

# 适应度函数
def fitness_continuous_coverage(chromosome):
    total_length = 0 # 测线总长度
    last_line = 0

    # 覆盖矩阵
    covering_matrix = np.zeros((251, 201))
    active_covering_matrix = np.zeros((251, 201))
    last_covering_matrix = active_covering_matrix


    for line in chromosome:
        # 覆盖宽度
        boundary_list = []
        for y in np.arange(0, length, 0.02):
            coverage_width, boundary_temp = cal_coverage(x=int((line/1852)//0.02), y=int(y//0.02), beta=0)
            # print(boundary_temp)

            # 更新覆盖矩阵
            for i in np.arange(boundary_temp[1][0], boundary_temp[1][1]+1):
                covering_matrix[i][int((line/1852)//0.02)] += 1
                active_covering_matrix[i][int((line/1852)//0.02)] += 1

            # boundary_list.append(boundary_temp)

        # 重复率
        if line == chromosome[0]:
            pass
        else:
            overlap = cal_overlap(active_covering_matrix, last_covering_matrix)
            print(overlap)
            # if overlap < 10 or overlap > 20:
            #     return 0  # 无效的解

        # 测线长度
        lenth = 5

        # 绘制矩阵图
        plt.imshow(active_covering_matrix, cmap='viridis')
        plt.colorbar()
        plt.show()
        
        total_length += lenth
        last_line = line

    # 覆盖率
    covering_percentage = np.count_nonzero(covering_matrix) / covering_matrix.size * 100
    print('覆盖率')
    print(covering_percentage)


    # 绘制矩阵图
    plt.imshow(covering_matrix, cmap='viridis')
    plt.colorbar()
    plt.show()

    # # 如果最后一个测线没有覆盖到东端，适应度为0
    # if last_line + coverage_width/2 < width:
    #     return 0

    return covering_percentage


# 多点交叉函数
def crossover(parent1, parent2):
    idx = np.random.randint(1, len(parent1)-1)
    child1 = np.concatenate((parent1[:idx], parent2[idx:]))
    child2 = np.concatenate((parent2[:idx], parent1[idx:]))
    return child1, child2

# 自适应变异函数
def adaptive_mutation(chromosome, fitness_value):
    if np.random.rand() < mutation_rate:
        idx = np.random.randint(0, len(chromosome))
        delta = np.random.uniform(-mutation_rate, mutation_rate) * (d_max - d_min)
        chromosome[idx] += delta


def cal_coverage(x, y, beta):
    '''
    求当前位置的覆盖宽度
    
    参数:
    x: int, 当前 x 坐标
    y: int, 当前 y 坐标
    beta: rad, 航行角

    返回值:
    float, 覆盖宽度
    list, 覆盖边界坐标点
    '''
    # 两点间距离公式
    def distance_between_points(x1, y1, x2, y2):
        x1, y1, x2, y2 = x1*0.02, y1*0.02, x2*0.02, y2*0.02
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    active_x = 0
    active_y = y
    active_z = elevation[y][x]  # 当前深度

    # 边界坐标
    boundary_temp = []

    # Left
    while True:
        active_x = x
        active_y += 1
        active_z = active_y * 0.02 * 1852 / math.tan(theta/2)
        # 比较扫描边界
        try:
            if active_z > elevation[active_y][active_x]:
                breadth_L = distance_between_points(x, y, active_x, active_y)
                boundary_temp.append((active_x, active_y))
                break
        except IndexError:
            print("IndexError!")
            if active_y < 0:
                breadth_L = distance_between_points(x, y, active_x, 0)
                boundary_temp.append((active_x, 0))
            elif active_y > 250:
                breadth_L = distance_between_points(x, y, active_x, 249)
                boundary_temp.append((active_x, 249))
            break
    
    active_x = 0
    active_y = y
    active_z = elevation[y][x]  # 当前深度

    # Right
    while True:
        active_x = x
        active_y += 1
        active_z = active_y * 0.02 * 1852 / math.tan(theta/2)

        # 比较扫描边界
        try:
            if active_z > elevation[active_y][active_x]:
                breadth_R = distance_between_points(x, y, active_x, active_y)
                boundary_temp.append((active_x, active_y))
                break
        except IndexError:
            print("IndexError!")
            if active_y < 0:
                breadth_R = distance_between_points(x, y, active_x, 0)
                boundary_temp.append((active_x, 0))
            elif active_y > 250:
                breadth_R = distance_between_points(x, y, active_x, 249)
                boundary_temp.append((active_x, 249))
            break

    breadth = breadth_L + breadth_R
    # print(x, y, boundary_temp)
    return breadth, boundary_temp


def cal_overlap(active_covering_matrix, last_covering_matrix):
    '''
    用于计算整条测线的均重复率

    参数:
    active_covering_matrix: matrix
    last_covering_matrix: matrix

    返回值:
    float, 重复率
    '''
    matching_indices = np.where((active_covering_matrix == 1) & (last_covering_matrix == 1))
    overlap = 100 * len(matching_indices[0]) / np.count_nonzero(active_covering_matrix)
    # print(len(matching_indices[0]), np.count_nonzero(active_covering_matrix))

    return overlap


# 初始化种群函数
def initialize_population():
    d_avg = (d_min + d_max) / 2
    num_lines = int(width*1852 / d_avg)
    print(num_lines)
    initial_population = []
    for _ in range(population_size):
        start = np.random.uniform(0, d_avg)
        chromosome = [start + i * d_avg for i in range(num_lines)]
        initial_population.append(chromosome)
    return np.array(initial_population)


def enhanced_genetic_algorithm_continuous_coverage():
    # 初始化种群
    population = initialize_population()

    print(population[0].tolist())
    
    # 遗传算法主循环
    for generation in range(num_generations):
        print(f'第{generation}代')

        # 评估适应度
        fitness_values = [fitness_continuous_coverage(chromo) for chromo in population]        

        # 精英策略: 选择最佳染色体
        elite_indices = np.argsort(fitness_values)[-elite_size:] # 按从小到大排序，取后几个
        new_population = [population[i] for i in elite_indices]
        
        # 交叉和变异
        while len(new_population) < population_size:
            # 选择父代
            parents = np.argsort(fitness_values)[-2:]

            # 交叉
            child1, child2 = crossover(population[parents[0]], population[parents[1]])
            new_population.extend([child1, child2])
            
            # 自适应变异
            for chromo in new_population[-2:]:
                adaptive_mutation(chromo, fitness_continuous_coverage(chromo))

        population = np.array(new_population)

    return population[np.argmax([fitness_continuous_coverage(chromo) for chromo in population])]


best_solution_continuous_coverage = enhanced_genetic_algorithm_continuous_coverage()
best_solution_list = best_solution_continuous_coverage.tolist()
print(best_solution_list)
