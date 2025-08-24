from functools import cmp_to_key
from typing import List, TypeVar
import random
import copy

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")

"""
.. module:: genetic_algorithm
   :platform: Unix, Windows
   :synopsis: Implementation of Genetic Algorithms.
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


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

        self.tabu = [0.0 for _ in range(10)]
        self.tabu_rounding = 3
        self.tabu_counter = 0
        self.max_steps = 5
        self.n_neighbors = 5
        self.step_size = 0.1

    def tabu_search(self, child: S):
        current = child
        best = child
        best_fitness = child.objectives[0]
        for _ in range(self.max_steps):
            neighbours = []
            for i in range(self.n_neighbors):
                variable = current.variables[0] + random.uniform(-self.step_size, self.step_size)
                n = copy.copy(current)
                n.variables[0] = variable
                self.problem.evaluate(n)
                neighbours.append(n)
            neighbours.sort(key=cmp_to_key(self.solution_comparator.compare))
            for n in neighbours:
                rounded = round(n.variables[0], self.tabu_rounding)
                if rounded not in self.tabu:
                    self.tabu.pop(0)
                    self.tabu.append(rounded)
                    current = n
                    break
                self.tabu_counter += 1
            else:
                current = neighbours[0]

            fitness = current.objectives[0]
            if fitness < best_fitness:
                best, best_fitness = current, fitness

        return best

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
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        child = offspring_population[0]
        return [self.tabu_search(child)]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        population.extend(offspring_population)

        population.sort(key=cmp_to_key(self.solution_comparator.compare))

        return population[: self.population_size]

    def result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return "Genetic algorithm"
