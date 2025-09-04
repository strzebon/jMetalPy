from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
import csv

if __name__ == "__main__":

    with open("results_per_eval.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Size", "Evaluations", "Fitness"])

        for n_problem in [1, 2, 10, 100]:
            problem = Rastrigin(n_problem)
            for _ in range(10):
                algorithm = GeneticAlgorithm(
                    problem=problem,
                    population_size=100,
                    offspring_population_size=1,
                    mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
                    crossover=SBXCrossover(0.9, 5.0),
                    termination_criterion=StoppingByEvaluations(max_evaluations=150000),
                )
                algorithm.run()
                for i, fitness in enumerate(algorithm.fitnesses):
                    writer.writerow([n_problem, (i+1) * 5000, fitness])