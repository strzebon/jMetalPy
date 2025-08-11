from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = Rastrigin(1)

    avg_fintness = 0
    for _ in range(100):
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

        print("Algorithm: {}".format(algorithm.get_name()))
        print("Problem: {}".format(problem.name()))
        print("Solution: {}".format(result.variables))
        print("Fitness: {}".format(result.objectives[0]))
        print("Computing time: {}".format(algorithm.total_computing_time))
        avg_fintness += result.objectives[0]
    print(f"Avg fitness:{avg_fintness/100}")
