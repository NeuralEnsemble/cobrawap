# from elephant.parallel import SingleProcess

# class parallelize:
#     """
#     Context manager that applies elephant.parallel executors:
#     ProcessPoolExecutor(), MPIPoolExecutor(), MPICommExecutor(), or
#     SingleProcess() (default).
#
#     Example:
#     ```
#     results = [my_function(arg) for arg in iterables_list]
#     ```
#     becomes
#     ```
#     with parallelize(my_function) as parallel_func:
#         results = parallel_func(iterables_list, *args, **kwargs)
#     ```
#     """
#     def __init__(self, func, executor=SingleProcess()):
#         self.executor = executor
#         self.func = func
#
#     def __enter__(self):
#
#         def _func(iterable, *args, **kwargs):
#
#             def arg_func(iterable):
#                 return self.func(iterable, *args, **kwargs)
#
#             result = self.executor.execute(arg_func, iterable)
#             return result
#
#         return _func
#
#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         pass
