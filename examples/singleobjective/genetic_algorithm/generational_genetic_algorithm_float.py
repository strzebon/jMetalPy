from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = Rastrigin(1)

    avg_fitness = 0
    for i in range(100):
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

        print("Fitness: {}".format(result.objectives[0]))
        print("Computing time: {}".format(algorithm.total_computing_time))
        print("Tabu list max: {}".format(max(algorithm.tabu.values())))
        print("Tabu counter: {}".format(algorithm.tabu_counter))
        print("Skip counter: {}".format(algorithm.skip_counter))
        avg_fitness += result.objectives[0]
    avg_fitness /= 100
    print("Average fitness: {}".format(avg_fitness))
