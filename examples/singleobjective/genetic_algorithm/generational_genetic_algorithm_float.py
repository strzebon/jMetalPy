from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
import csv

if __name__ == "__main__":

    with open("results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Size", "Fitness", "Time"])
        n_problem = 1
        while n_problem <= 128:
            problem = Rastrigin(n_problem)
            avg_fitness = 0
            for i in range(10):
                algorithm = GeneticAlgorithm(
                    problem=problem,
                    population_size=100,
                    offspring_population_size=1,
                    mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
                    crossover=SBXCrossover(0.9, 5.0),
                    termination_criterion=StoppingByEvaluations(max_evaluations=150000),
                )
                algorithm.run()
                result = algorithm.result()

                computing_time = algorithm.total_computing_time

                print("Fitness: {}".format(result.objectives[0]))
                print("Computing time: {}".format(computing_time))
                print("Tabu counter: {}".format(algorithm.tabu_counter))
                print("Skip counter: {}".format(algorithm.skip_counter))
                fitness = result.objectives[0]
                avg_fitness += fitness
                writer.writerow([n_problem, fitness, computing_time])
            avg_fitness /= 10
            print("Average fitness: {}".format(avg_fitness))
            n_problem *= 2
