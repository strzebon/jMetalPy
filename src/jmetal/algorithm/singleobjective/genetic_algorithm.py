from functools import cmp_to_key
from typing import List, TypeVar

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion
from decimal import Decimal

S = TypeVar("S")
R = TypeVar("R")

"""
.. module:: genetic_algorithm
   :platform: Unix, Windows
   :synopsis: Implementation of Genetic Algorithms.
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def leading_zeros_after_decimal(num: int) -> int:
    d = Decimal(str(num)).normalize()
    s = str(d)

    if "." not in s:
        return 0

    frac = s.split(".")[1]
    count = 1
    for ch in frac:
        if ch == "0":
            count += 1
        else:
            break
    return count


class GeneticAlgorithm(EvolutionaryAlgorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
            solution_comparator: Comparator = ObjectiveComparator(0)
    ):
        super(GeneticAlgorithm, self).__init__(
            problem=problem, population_size=population_size, offspring_population_size=offspring_population_size
        )
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.solution_comparator = solution_comparator

        self.selection_operator = selection

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.mating_pool_size = (
            self.offspring_population_size
            * self.crossover_operator.get_number_of_parents()
            // self.crossover_operator.get_number_of_children()
        )

        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

        self.lower_bound = -5.12
        self.upper_bound = 5.12
        self.n_buckets = 10**2
        self.avg = 0
        self.tabu = {}
        self.tabu_threshold = 1.0
        self.pheromone_boost = 1.0
        self.evaporation_rate = 0.1
        self.tabu_counter = 0
        self.skip_counter = 0

    def bin_vector(self, solution: S):
        return int((solution.variables[0] - self.lower_bound) // ((self.upper_bound - self.lower_bound) / self.n_buckets))

    def calculate_pheromones(self, population: List[S]):
        for key in list(self.tabu.keys()):
            self.tabu[key] *= (1 - self.evaporation_rate)

        for solution in population:
            key = self.bin_vector(solution)
            if key not in self.tabu:
                self.tabu[key] = 0.0
            self.tabu[key] += self.pheromone_boost

    def create_initial_solutions(self) -> List[S]:
        return [self.population_generator.new(self.problem) for _ in range(self.population_size)]

    def evaluate(self, population: List[S]):
        return self.population_evaluator.evaluate(population, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def selection(self, population: List[S]):
        mating_population = []

        for _ in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception("Wrong number of parents")

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                key = self.bin_vector(solution)
                if key in self.tabu and self.tabu[key] > self.tabu_threshold:
                    self.tabu_counter += 1
                    continue
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break
            else:
                self.skip_counter += 1

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population.extend(offspring_population)

        population.sort(key=cmp_to_key(self.solution_comparator.compare))

        self.calculate_pheromones(population[: self.population_size])

        if self.evaluations % 10_000 == 0:
            self.avg = 0
            for solution in population[: self.population_size]:
                self.avg += solution.variables[0]
            self.avg /= self.population_size
            print("Average variable: {}".format(self.avg))
            num_of_zeros = leading_zeros_after_decimal(self.avg)
            if 10 ** (num_of_zeros + 3) > self.n_buckets:
                self.n_buckets = 10 ** (num_of_zeros + 3)
                print("change nubmer of buckets: {}".format(self.n_buckets))
                self.tabu = {}


        return population[: self.population_size]

    def result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return "Genetic algorithm"
