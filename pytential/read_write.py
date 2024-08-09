import dill
from os.path import getmtime, isfile, dirname, join, basename, splitext
from importlib import import_module, util

def write_pytential(pytential, file_name):
    dill.settings['recurse'] = True
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    with open(file_name, 'wb') as output:
        dill.dump(pytential, output)

def read_pytential(file_name):
    if not name.endswith('.pkl'):
        name += '.pkl'
    with open(name, 'rb') as input:
         pytential = dill.load(input)
    return pytential

# def load_pytential(file_name):
    
#     def build_pytential_from_file_path(file_path):
#         mod_name = splitext(basename(file_path))[0]
#         spec = util.spec_from_file_location(mod_name, file_path)
#         pytential_file = util.module_from_spec(spec)
#         spec.loader.exec_module(pytential_file)
#         print('Building pytential')
#         return pytential_file.build_pytential()

#     def load_or_build_pytential_from_file(file_name):
#         file_name_py = file_name +'.py'
#         file_name_saved = file_name +'.pkl'

#         # Handling mpi distribution: pytentials are loaded by all ranks, but are built on one rank.
#         # Not ideal since it implies copies of pytentials everywhere. Better to centralize...?

#         # Ensure the saved pytential is up to date.
#         if isfile(file_name_py):
#             if not isfile(file_name_saved) or getmtime(file_name_py) > getmtime(file_name_saved):
#                 pytential = build_pytential_from_file_path(file_name_py)
#                 pytential.write_to_file(file_name_saved)

#         #---->Check to ensure pytential only being built on one rank. Needed?
#         # from mpi4py import MPI
#         # comm = MPI.COMM_WORLD
#         # rank = comm.Get_rank()
#         # if rank ==0:
#         #     if isfile(file_name_py):
#         #         if not isfile(file_name_saved) or getmtime(file_name_py) > getmtime(file_name_saved):
#         #             pytential = build_pytential_from_file_path(file_name_py)
#         #             pytential.write_to_file(file_name_saved)
#         # comm.barrier()
        
#         if isfile(file_name_saved):
#             return read_pytential(file_name_saved)


#     module_path = dirname(__file__) if '__file__' in globals() else '.'
#     module_path = join(module_path, 'common_systems')
#     paths = ['.', module_path]

#     for path in paths:
#         file_path = join(path, file_name)
#         pytential = load_or_build_pytential_from_file(file_path)
#         if pytential is not None:
#             return pytential

#     raise FileNotFoundError("pytential not found")

