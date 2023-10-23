import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)

def load_data_from_file(fileName = "advertising.csv"):
    data = np.genfromtxt(fileName, dtype=None, delimiter=',', skip_header=1)
    features_X = data[:, :3]
    sales_Y = data[:, 3]

    intercept = np.ones((features_X.shape[0], 1))
    features_X = np.concatenate((intercept, features_X), axis=1)
    return features_X, sales_Y


features_X, sales_Y = load_data_from_file()


def generate_random_value(bound = 10):
    return (random.random() - 0.5)*bound


def create_individual(n=4, bound=10):
    individual = [generate_random_value() for _ in range(n)]
    return individual


def compute_loss(individual):
    theta = np.array(individual)
    y_hat = features_X.dot(theta)
    loss  = np.multiply((y_hat-sales_Y), (y_hat-sales_Y)).mean()
    return loss


def compute_fitness(individual):
    loss = compute_loss(individual)
    fitness = 1 / (loss + 1)
    return fitness


def crossover(individual1, individual2, crossover_rate = 0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

    for i in range(len(individual1)):
        if random.random() < crossover_rate:
            individual1_new[i] = individual2[i]
            individual2_new[i] = individual1[i]

    return individual1_new, individual2_new


def mutate(individual, mutation_rate = 0.05):
    individual_m = individual.copy()

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual_m[i] = generate_random_value()

    return individual_m


def initializePopulation(m):
  population = [create_individual() for _ in range(m)]
  return population


def selection(sorted_old_population, m):
    index1 = random.randint(0, m-1)
    while True:
        index2 = random.randint(0, m-1)
        if (index2 != index1):
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s


def create_new_population(old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=compute_fitness)

    if gen%1 == 0:
        print("Best loss:", compute_loss(sorted_population[m-1]), "with chromsome: ", sorted_population[m-1])

    new_population = []
    while len(new_population) < m-elitism:
        # selection
        individual_s1 = selection(sorted_population, m)
        individual_s2 = selection(sorted_population, m) # duplication

        # crossover
        individual_t1, individual_t2 = crossover(individual_s1, individual_s2)

        # mutation
        individual_m1 = mutate(individual_t1)
        individual_m2 = mutate(individual_t2)

        new_population.append(individual_m1)
        new_population.append(individual_m2)

    for ind in sorted_population[m-elitism:]:
        new_population.append(ind.copy())

    return new_population, compute_loss(sorted_population[m-1])


def run_GA():
    n_generations = 100
    m = 600
    features_X, sales_Y = load_data_from_file()
    population = initializePopulation(m)
    losses_list = []
    for i in range(n_generations):
        population, losses = create_new_population(population, 2, i)
        losses_list.append(losses)
    return population, losses_list


def visualize_loss(losses_list):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(losses_list, c='green')
    plt.xlabel('Generations')
    plt.ylabel('Losses')
    fig.savefig('Loss.jpg')


def visualize_predict_gt_line(population):
    # visualization of ground truth and predict value
    sorted_population = sorted(population, key=compute_fitness)
    print(sorted_population[-1])
    theta = np.array(sorted_population[-1])

    estimated_prices = []
    for feature in features_X:
        estimated_price = sum(c*x for x, c in zip(feature, theta))
        estimated_prices.append(estimated_price)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.plot(sales_Y, c='green', label='Real Prices')
    plt.plot(estimated_prices, c='blue', label='Estimated Prices')
    plt.legend()
    fig.savefig('Visualization_Line.jpg')


def visualize_predict_gt_scatter(population):
    # visualization of ground truth and predict value
    sorted_population = sorted(population, key=compute_fitness)
    print(sorted_population[-1])
    theta = np.array(sorted_population[-1])

    estimated_prices = []
    samples = [i for i in range(len(features_X))]
    for feature in features_X:
        estimated_price = sum(c*x for x, c in zip(feature, theta))
        estimated_prices.append(estimated_price)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.scatter(samples, sales_Y, c='green', label='Real Prices')
    plt.scatter(samples, estimated_prices, c='blue', label='Estimated Prices')
    plt.legend()
    fig.savefig('Visualization_Scatter.jpg')


if __name__ == "__main__":
    population, losses_list = run_GA()
    visualize_loss(losses_list)
    visualize_predict_gt_line(population)
    visualize_predict_gt_scatter(population)