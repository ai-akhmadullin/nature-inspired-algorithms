import argparse
import math
import functools
import copy
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from xml.dom import minidom
from collections import namedtuple

# Command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data", default='Input_data/data_32.xml', type=str, help="Path to the data file")

# Namedtuples for representing data
Node = namedtuple('Node', ['id', 'type', 'x', 'y'])
Vehicle = namedtuple('Vehicle', ['departure', 'arrival', 'capacity'])
Request = namedtuple('Request', ['id', 'node', 'quantity'])

def parse_input(file):
    # Parse the nodes
    xml_nodes = file.getElementsByTagName('node')
    nodes = {}
    for node in xml_nodes:
        id = int(node.attributes['id'].value)
        type = int(node.attributes['type'].value)
        x = float(node.childNodes[1].firstChild.data)
        y = float(node.childNodes[3].firstChild.data)
        nodes[id] = Node(id, type, x, y)

    # Parse the vehicle
    xml_vehicle = file.getElementsByTagName('vehicle_profile')[0]
    departure = int(xml_vehicle.childNodes[1].firstChild.data)
    arrival = int(xml_vehicle.childNodes[3].firstChild.data)
    capacity = float(xml_vehicle.childNodes[5].firstChild.data)
    vehicle = Vehicle(departure, arrival, capacity)

    # Parse the requests
    xml_requests = file.getElementsByTagName('request')
    requests = []
    for request in xml_requests:
        id = int(request.attributes['id'].value)
        node = int(request.attributes['node'].value)
        quantity = float(request.childNodes[1].firstChild.data)
        requests.append(Request(id, node, quantity))

    return nodes, vehicle, requests

# Method for computing the Euclidean distance between two nodes
@functools.lru_cache(maxsize=None)
def distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

# Method for generating `num_ants` solutions based on pheromones
def generate_solutions(nodes, vehicle, requests, pheromones, num_ants, alpha=1, beta=1.5):
    # Method for computing the probability of going from `node1` to `node2` based on distance and pheromones.
    # `alpha` is a weight parameter for pheromones and `beta` is a weight parameter for distance
    def compute_prob(avail_capacity, node1, node2):
        # If we cannot complete this request due to the lack of capacity, return 0
        if avail_capacity < node2.quantity:
            return 0
    
        dist = 1/distance(node1, nodes[node2.node])
        tau = pheromones[node1.id-1, node2.node-1]
        ret = pow(tau, alpha) * pow(dist,beta)
        return ret if ret > 0.000001 else 0.000001
    
    # Generate `num_ants` solutions
    for ant in range(num_ants):
        route = [nodes[vehicle.departure]]
        req = copy.deepcopy(requests)
        capacity = vehicle.capacity
        while req:
            # Build an array of probabilities for choosing each request
            probs = np.array(
                [compute_prob(capacity, route[-1], node) for node in req]
            )
            # There're some probabilities != 0, which means we have enough capacity to complete some request
            if np.any(probs):
                next_node = req[np.random.choice(np.arange(len(req)), p=probs/sum(probs))]
                route.append(nodes[next_node.node])
                req.remove(next_node)
                capacity -= next_node.quantity
            # There's not enough capacity to complete any request, go to the depot
            else:
                route.append(nodes[vehicle.departure])
                capacity = vehicle.capacity
        route.append(nodes[vehicle.arrival])
        yield route

# Method for updating pheromone matrix based on solutions and their fitness values
def update_pheromones(pheromones, solutions, fits, Q=100, rho=0.6):
    pheromone_update = np.zeros(shape=pheromones.shape)
    for solution, fit in zip(solutions, fits):
        for node1, node2 in zip(solution, solution[1:]):
            pheromone_update[node1.id-1][node2.id-1] += Q/fit
    
    return (1-rho)*pheromones + pheromone_update

# Solve VRP problem
def vrp_solver(nodes, vehicle, requests, num_iterations=800, num_ants=30, random_seed=42):
    # Local method for calculating the fitness value of a solution
    def calc_fitness(solution):
        solution_distance = 0
        for node1, node2 in zip(solution, solution[1:]):
            solution_distance += distance(node1, node2)
        return solution_distance

    # Throughout the program we build the route as one array for convenience. When we are done, it is more representative to
    # divide the route so that each array inside the `divided_route` represents a route for one car
    def divide_route(solution):
        divided_route = []
        car = [solution[0]]
        for i in range(1, len(solution)):
            car.append(solution[i])
            if solution[i].type == 0:
                divided_route.append(car)
                car = [solution[0]]
        return divided_route

    # Fix the random seed
    np.random.seed(random_seed)
    
    # Create a pheromone matrix. Pheromone[x][y] is the pheromone trail from x-th node to y-th node
    pheromones = np.ones((len(nodes), len(nodes))) * 0.01
    
    best_solution = None
    best_fitness = float('inf')
    history = np.zeros(num_iterations)
    
    # At one iteration, create `num_ants` solutions, evaluate their fitness values and update the pheromone matrix
    for i in range(num_iterations):
        solutions = list(generate_solutions(nodes, vehicle, requests, pheromones, num_ants))
        fits = [calc_fitness(sol) for sol in solutions]
        pheromones = update_pheromones(pheromones, solutions, fits)
        
        for solution, fitness in zip(solutions, fits):
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution
        history[i] = best_fitness
        
    best_solution = divide_route(best_solution)
    return best_solution, best_fitness, pheromones, history

# Display the results: best solution, its distance, the number of cars we need.
# Draw the route for each car with different colors
def show_results(nodes, pheromones, solution, fitness, history):
    # Plot the history of best fitness values
    plt.plot(history)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Progress')
    plt.show()
    
    # Draw the grid
    lines = []
    colors = []
    list_nodes = sorted(nodes.values(), key=lambda node: node.id)
    for i, node1 in enumerate(list_nodes):
        for j, node2 in enumerate(list_nodes):
            lines.append([(node1.x, node1.y), (node2.x, node2.y)])
            colors.append(pheromones[i][j])

    lc = mc.LineCollection(lines, linewidths=np.array(colors))

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()

    # Print the results
    num_cars = len(solution)
    print('Distance: ', fitness)
    print('Number of vehicles: ', num_cars)

    # Represent one route with one color
    cmap = cm.get_cmap('tab10', num_cars)  
    for i, car in enumerate(solution):
        solution_lines = []
        for node1, node2 in zip(car, car[1:]):
            solution_lines.append([(node1.x, node1.y), (node2.x, node2.y)])
        color = cmap(i)  
        solutions_lc = mc.LineCollection(solution_lines, colors=color)
        ax.add_collection(solutions_lc)

    plt.show()


def main(args):
    # Parse the XML input
    file = minidom.parse(args.data)
    nodes, vehicle, requests = parse_input(file)
    
    # Find the best solution
    best_solution, dist, pheromones, history = vrp_solver(nodes, vehicle, requests)
    
    # Show the results
    show_results(nodes, pheromones, best_solution, dist, history)



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)