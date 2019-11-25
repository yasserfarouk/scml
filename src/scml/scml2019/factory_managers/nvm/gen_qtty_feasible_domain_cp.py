# from constraint import *
# import time
# import pandas as pd
# import os
#
#
# def set_to_zero_constraint(x):
#     """
#     A constraint that sets a variable to zero.
#     :param x:
#     :return:
#     """
#     return x == 0
#
#
# def storage_constraint(*args):
#     """
#     Constraints the sum of the first half of the variables to be <= the sum of the second half of the variables.
#     Used to implement the storage constraint and the production constraint.
#     :param args:
#     :return:
#     """
#     middle = int(len(args) / 2) + 1
#     return args[0] <= sum([args[h] for h in range(1, middle)]) - sum([args[h] for h in range(middle, len(args))])
#
#
# def first_sum_leq_second_sum(*args):
#     """
#     Make sure we don't endup with more inputs than outputs
#     :param args:
#     :return:
#     """
#     middle = int(len(args) / 2)
#     return sum([args[h] for h in range(middle)]) <= sum([args[h] for h in range(middle, len(args))])
#
#
# def solve_cp(the_production_capacity: int, the_time_periods: int, the_qtty_domain_size: int) -> dict:
#     """
#     Create the CP problem.
#     :param the_production_capacity:
#     :param the_time_periods:
#     :param the_qtty_domain_size:
#     :return:
#     """
#     # Create the problem
#     problem = Problem()
#
#     # Create the Variables
#     for t in range(the_time_periods):
#         problem.addVariable("x_" + str(t), [q for q in range(the_qtty_domain_size)])
#         problem.addVariable("y_" + str(t), [q for q in range(the_qtty_domain_size)])
#         # The domain of the production variable is limited to the production capacity.
#         problem.addVariable("z_" + str(t), [q for q in range(the_production_capacity + 1)])
#
#     # Create the constraints
#
#     # (0) Initial day Constraints
#     problem.addConstraint(set_to_zero_constraint, variables=["y_0"])
#     problem.addConstraint(set_to_zero_constraint, variables=["z_0"])
#
#     for t in range(1, the_time_periods):
#         # (1) Storage Constraints
#         problem.addConstraint(storage_constraint, variables=["y_" + str(t)] + ["z_" + str(t_inner) for t_inner in range(t)] + ["y_" + str(t_inner) for t_inner in range(t)])
#         problem.addConstraint(storage_constraint, variables=["z_" + str(t)] + ["x_" + str(t_inner) for t_inner in range(t)] + ["z_" + str(t_inner) for t_inner in range(t)])
#
#     # -- From this point on, we implement heuristic constraints
#     # (2) The total number of inputs is no more than the total number of outputs.
#     problem.addConstraint(first_sum_leq_second_sum,
#                           variables=["x_" + str(t_inner) for t_inner in range(the_time_periods)] + ["y_" + str(t_inner) for t_inner in range(the_time_periods)])
#
#     # (3) The total number of production is no more than the total number of outputs.
#     problem.addConstraint(first_sum_leq_second_sum,
#                           variables=["z_" + str(t_inner) for t_inner in range(the_time_periods)] + ["y_" + str(t_inner) for t_inner in range(the_time_periods)])
#
#     return problem.getSolutions()
#
#
# def get_zip_location(the_time_periods: int, the_qtty_domain_size) -> str:
#     """
#     Produces the location of the zip file as a string
#     :param the_time_periods:
#     :param the_qtty_domain_size:
#     :return:
#     """
#     return 'qtty_domain/qtty_domain_t_' + str(the_time_periods) + '_d_' + str(the_qtty_domain_size) + '.zip'
#
#
# def generate_and_save_sol_df(the_time_periods: int, the_qtty_domain_size: int, the_sols: dict):
#     data = []
#     for sol in the_sols:
#         d = [[sol['x_' + str(t)], sol['y_' + str(t)], sol['z_' + str(t)]] for t in range(the_time_periods)]
#         flat_list = [item for sublist in d for item in sublist]
#         data += [flat_list]
#     c = [['x_' + str(t), 'y_' + str(t), 'z_' + str(t)] for t in range(the_time_periods)]
#     d = pd.DataFrame(data, columns=[item for sublist in c for item in sublist])
#     d.to_csv(get_zip_location(the_time_periods, the_qtty_domain_size), index=False, compression='gzip')
#
#
# production_capacity = 10
# # Solve the CP
# for number_of_periods in range(1, 7):
#     for quantities_domain_size in range(2, 21):
#         if not os.path.isfile(get_zip_location(number_of_periods, quantities_domain_size)):
#             t0_generation = time.time()
#             sols = solve_cp(production_capacity, number_of_periods, quantities_domain_size)
#             print(f'done generating solutions for t = {number_of_periods}, '
#                   f'Domain(q) = {quantities_domain_size}. '
#                   f'took ' + format(time.time() - t0_generation, '.6f') + ' secs. ', end='')
#
#             t0_saving_data = time.time()
#             generate_and_save_sol_df(number_of_periods, quantities_domain_size, sols)
#             print(f'done saving compressed dataframe. took ' + format(time.time() - t0_saving_data, '.6f') + ' seconds')
#         else:
#             print(f'Already have the feasible domain for t = {number_of_periods} and Domain(q) = {quantities_domain_size}')
