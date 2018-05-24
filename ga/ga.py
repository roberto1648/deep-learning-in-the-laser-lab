import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import copy
import time
import traceback
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import DBAdapters
from pyevolve import Selectors
from pyevolve import Statistics
from pyevolve import Consts
from pyevolve import Scaling
from pyevolve import Initializators
from pyevolve import Mutators
import utils


def sum_fitness(individual,
                save_in = "",
                graph = False):
    """
    this is intended as simple example of the fitness function. the individual
    is a vector whose elements are between 0 and 1. there can also be two
    optional if statement blocks for graph (a boolean) and save_in (a string,
    which is ignored if it's empty.)
    """
    import time
    # time.sleep(0.1)
    if graph:
        yy = np.array(individual)
        xx = np.arange(0, np.size(yy))
        plt.plot(xx, yy)
    if save_in: # this is called by the ga at the end
        import shelve
        db = shelve.open(save_in)
        db["saving example"] = "save something"
        db.close()
    # return sum(np.mod(individual, 1))
    return np.sum(individual)


def main(fitness_function=sum_fitness,
         population_size=10,
         parameters_per_individual=160,
         parameter_bounds=(-1, 1),
         mutation_rate=0.02,
         crossover_rate=0.8,
         freq_stats=50,
         max_gens=2000,
         callback_functions=[],
         optimization_type='maximize',
         temp_fname='.__fitness_history__.csv',
         stop_fname='.stop'):
    """
    run the ga
    """
    genome = getGenome(fitness_function, parameters_per_individual,
              parameter_bounds)
    # genome = G1DList.G1DList(parameters_per_individual)
    # genome.initializator.set(Initializators.G1DListInitializatorReal)
    # genome.evaluator.set(fitness_function)
    # genome.setParams(rangemin = parameter_bounds[0],
    #                  rangemax = parameter_bounds[1])

    ga = setupGAEngine(genome=genome,
                  population_size=population_size,
                  mutation_rate=mutation_rate,
                  crossover_rate=crossover_rate,
                  max_gens=max_gens,
                  callback_functions=callback_functions,
                  optimization_type=optimization_type,
                  temp_fname=temp_fname,
                  stop_fname=stop_fname)
    # ga = GSimpleGA.GSimpleGA(genome)
    # ga.setPopulationSize(population_size)
    # ga.setMinimax(Consts.minimaxType['maximize'])
    # # ga.setCrossoverRate(crossover_probability)
    # # ga.setMutationRate(mutation_probability)
    # adapter = DBAdapters.DBFileCSV(filename="file.csv", identify="run_01",
    #                                frequency=1, reset=True)
    # ga.setDBAdapter(adapter)
    # ga.selector.set(Selectors.GRankSelector)
    # ga.setGenerations(maxgens)
    # ga.terminationCriteria.set(GSimpleGA.RawStatsCriteria)
#
    # pop = ga.getPopulation()
    # pop.scaleMethod.set(Scaling.SigmaTruncScaling)

    ga.evolve(freq_stats=freq_stats)

    # print ga.bestIndividual()[::10]

    return ga


def getGenome(fitness_function=sum_fitness,
              parameters_per_individual=160,
              parameter_bounds=(-1, 1)):
    genome = G1DList.G1DList(parameters_per_individual)
    genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.evaluator.set(fitness_function)
    genome.setParams(rangemin=parameter_bounds[0],
                     rangemax=parameter_bounds[1])
    return genome


def setupGAEngine(genome=G1DList.G1DList(),
                  population_size=10,
                  mutation_rate=0.02,
                  crossover_rate=0.8,
                  max_gens=2000,
                  callback_functions=[],
                  optimization_type='maximize',
                  temp_fname='.__fitness_history__.csv',
                  stop_fname='.stop'):
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setPopulationSize(population_size)
    ga.setMinimax(Consts.minimaxType[optimization_type])
    # ga.setCrossoverRate(crossover_probability)
    genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    # genome.initializator.set()
    # genome.mutator.
    # ga.setMutationRate(mutation_probability)

    fname = os.path.join(ga_folder(), temp_fname)
    utils.delete_file(fname)
    ga.stepCallback.set(lambda x: saveGenerationStats(x, temp_fname))
    # ga.stepCallback.add(mockCallback)

    for func in callback_functions:
        ga.stepCallback.add(func)
        # fun = lambda x: callbackWrapper(x, callback_function)
        # def funcWrap(x):
        #     __ = func()
        #     return False
        # ga.stepCallback.add(copy.deepcopy(funcWrap))

    # fname = os.path.join(ga_folder(), temp_fname)
    # adapter = DBAdapters.DBFileCSV(filename=fname,
    #                                identify="run_01",
    #                                frequency=1, reset=True)
    # ga.setDBAdapter(adapter)

    ga.selector.set(Selectors.GRankSelector)
    ga.setGenerations(max_gens)
    ga.setMutationRate(mutation_rate)
    ga.setCrossoverRate(crossover_rate)

    fname = os.path.join(ga_folder(), stop_fname)
    utils.delete_file(fname)
    ga.terminationCriteria.set(lambda x: checkStop(x, stop_fname))

    pop = ga.getPopulation()
    pop.scaleMethod.set(Scaling.SigmaTruncScaling);print len(pop)
    # pop[0] = population_size * [0]

    return ga


# def callbackWrapper(ga_engine=GSimpleGA.GSimpleGA(G1DList.G1DList()),
#                     callback_function=sum_fitness):
#     __ = callback_function()
#     return False


def checkStop(ga_engine=GSimpleGA.GSimpleGA(G1DList.G1DList()),
              stop_fname='.stop'):
    condition_1 = GSimpleGA.ConvergenceCriteria(ga_engine)
    fname = os.path.join(ga_folder(), stop_fname)
    condition_2 = os.path.exists(fname)
    return condition_1 or condition_2


def saveGenerationStats(ga_engine=GSimpleGA.GSimpleGA(G1DList.G1DList()),
                        temp_fname='.__fitness_history__.csv'):
    """
    saves the generation stats to temp_fname.
    Intended to be run after each generation. Needs to be attached to
    the ga engine, like so: ga.stepCallback.set(saveGenerationStats) (if
    no other callback is already attached)
    or ga.stepCallback.add(saveGenerationStats)
    :param ga_engine:
    :return: False (if True it will stop the evolution)
    """
    fname = os.path.join(ga_folder(), temp_fname)
    stats = ga_engine.getStatistics()
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df = df.append(pd.Series(stats.internalDict), ignore_index=True)
        df.to_csv(fname, index=False)
    else:
        df = pd.DataFrame(data = [stats.internalDict])
        df.to_csv(fname, index=False)
    return False


# def MyG1DListInitializatorRealRandom(genome, **args):
#
#     """ Real initialization function of G1DList
#
#     This initializator accepts the *rangemin* and *rangemax* genome parameters.
#     Uses the function createRandomChromosomeBoundary
#
#     """
#
#     range_min = genome.getParam("rangemin", 0 )
#
#     range_max = genome.getParam("rangemax", 1 )
#
#     garegion = genome.getParam("garegion")
#
#     n = (genome.genomeSize)
#
#     genome.genomeList = garegion.createMyRandomChromosomeBoundary()


# def insert_individual_at_generation0(ga_engine=GSimpleGA.GSimpleGA(G1DList.G1DList()),
#                                      individual=[]):
#     generation = ga_engine.getCurrentGeneration()
#
#     if (generation == 0) and (individual != []):
#         pop = ga_engine.getPopulation()
#         pop


# def mockCallback(*args, **kwargs):
#     print 'bye'
#     return False


def save(fname="ga_run",
         ga_engine=GSimpleGA.GSimpleGA(G1DList.G1DList()),
         fitness_function=sum_fitness,
         fitness_history_fname='.__fitness_history__.csv'
         ):
    # copy the temp file:
    import shutil

    extensionless_fname, extension = os.path.splitext(fname)

    temp_fname = os.path.join(ga_folder(), fitness_history_fname)
    new_file = extensionless_fname + '__fitness_history.csv'
    shutil.copy(temp_fname, new_file)
    print 'fitness history saved to "{}"'.format(new_file)

    # best_ind = [x for x in ga_engine.bestIndividual()]
    best_ind = ga_engine.bestIndividual()
    pars = np.array(best_ind.genomeList)
    new_file = extensionless_fname + "__parameters.txt"
    np.savetxt(new_file, pars)
    print 'best parameters saved to "{}"'.format(new_file)

    # save the best parameters translated by the fitness function:
    # if the fitness function can save, do it for the
    # best parameters:
    try:
        new_file = extensionless_fname + "__best" + extension
        fitness_function(pars, save_in=new_file)
        print 'best parameters translated by "{}" saved to "{}"'.format(fitness_function,
                                                                    new_file)
    except:
        traceback.print_exc()


def plotBest(ga_engine=GSimpleGA.GSimpleGA(G1DList.G1DList())):
    best_ind = ga_engine.bestIndividual()
    pars = np.array(best_ind.genomeList)
    pmin, pmax = best_ind.getParam('rangemin'), best_ind.getParam('rangemax')

    plt.figure('best parameters')
    plt.plot(pars)
    plt.xlabel('parameter index')
    plt.ylabel('parameter value in range [{}, {}]'.format(pmin, pmax))

    try:
        fitness_function = best_ind.evaluator[0]
        __ = fitness_function(pars, graph=True)
    except:
        print "couldn't access plot from fitness function '{}'".format(fitness_function)


def plotFitnessHistory(file_name='.__fitness_history__.csv'):
    if os.path.exists(file_name):
        fname = file_name
    else:
        fname = os.path.join(ga_folder(), file_name)

    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df.plot()
    else:
        print 'could not find file named {}'.format(fname)


#todo: to view current measurement. must see if it won't slow down
#too much the calculations.
# def saveCurrentMeasurement(current_fitness=-1,
#                              current_parameters=[],
#                              tdelay=2.,
#                              file_name='.__current_measurement__.csv'):
#     fname = os.path.join(utils.ga_folder(), file_name)
#     if os.path.exists(fname):
#         tlastmod = os.stat(fname).st_mtime
#     else:
#         tlastmod = 0
#
#     # write to file if it was modified more than tdelay seconds ago
#     if time.time() - tlastmod > tdelay:
#         s = pd.Series(data=[current_fitness,
#                             current_parameters],
#                       index=['fitness', 'parameters'])
#         s.to_csv(fname)
#
#
# def fitnessWrapper(X,
#                     fitness_function=sum_fitness,
#                     tdelay=2.,
#                     file_name='.__current_measurement__.csv',
#                     **kwargs
#                     ):
#     pars = [x for x in X]
#     fitness = fitness_function(pars, **kwargs)
#     saveCurrentMeasurement(fitness, pars,
#                            tdelay, file_name)


def ga_folder():
    # import sys
    # import os
    this_file_path = sys.modules[__name__].__file__
    return os.path.dirname(this_file_path)
