import concurrent.futures
import numpy as np
import pickle
import random
import string

import SharedArray

__version__ = '0.2.0'

_array_wrappers = dict()


def register_array_wrapper(cls, attrname):
    _array_wrappers[cls] = attrname


def dtype_to_tuple(dtype):
    """Return list of tuples needed to construct the dtype

    dtype == np.dtype(unpack_dtype(dtype))
    """
    result = []
    fields = dtype.fields
    for field_name in dtype.names:
        fieldinfo = fields[field_name]
        if len(fieldinfo) == 3:
            # The field has a "titles" attribute.
            # In this case, the tuple returned by .fields is inconsistent
            # with the tuple expected by np.dtype constructor :-(
            field_dtype, some_number, field_title = fieldinfo
            result.append(((field_title, field_name), field_dtype))
        else:
            field_dtype, some_number = fieldinfo
            result.append((field_name, field_dtype))
    return result


def pickle_as_array(x):
    """Pickle x and view its bytes as a numpy array"""
    return np.frombuffer(pickle.dumps(x),
                         dtype=np.uint8)


def unpickle_from_array(x):
    """Inverse of pickle_as_array"""
    # Pickle will use the buffer protocol on x automatically
    return pickle.loads(x)


def pack_dtype(dtype):
    """Pack dtype into a numpy array"""
    return pickle_as_array(dtype_to_tuple(dtype))


def unpack_dtype(x):
    """Unpack dtype from a numpy array"""
    return np.dtype(unpickle_from_array(x))


def shm_clear():
    """Clear all npshmex-associated shared memory"""
    for x in SharedArray.list():
        x = x.name.decode()
        if x.startswith('npshmex'):
            SharedArray.delete(x)


def is_shmable(thing):
    """Return whether thing can be passed along through shared memory"""
    if isinstance(thing, np.ndarray):
        return True
    for x in _array_wrappers:
        if isinstance(thing, x):
            return True
    return False


def shm_put(arr, temp=False, _key=None, _is_wrapper=False):
    """Put array into shared memory and return key

    :param temp: If True, key will be deleted on its first retrieval by shm_pop
    (unless shm_pop is called with keep=True).
    :param _key: shared memory key to use. Do not set this yourself
    unless you know what you are doing.
    """
    if not is_shmable(arr):
        raise ValueError(f"Got {arr} ({type(arr)}) instead of a numpy array "
                         f"or registered array wrapper!")

    # Deal with array-wrappers
    for x, fieldname in _array_wrappers.items():
        if isinstance(arr, x):
            # Put data on shared memory
            data = getattr(arr, fieldname)
            datakey = shm_put(data)
            # Replace data by the SHM key, and pickle it
            setattr(arr, fieldname, datakey)
            return shm_put(pickle_as_array(arr),
                           _is_wrapper=True)

    is_struct = arr.dtype.names is not None

    if _key is None:
        _key = 'shm://npshmex_' + '_'.join([
            'wrapper' if _is_wrapper else ("sdata" if is_struct else "data"),
            ''.join(random.choices(string.ascii_uppercase
                                   + string.digits,
                                   k=16)),
            "TEMP" if temp else ""])

    shared_arr = SharedArray.create(_key, arr.shape, dtype=arr.dtype)

    if is_struct:
        # Structured arrays can only be shared as raw void-dtype arrays
        shared_arr[:] = np.frombuffer(arr, dtype=np.void(arr.itemsize))

        # To recover the dtype later, we have to encode it as a separate array
        # and share that too
        shm_put(pack_dtype(arr.dtype),
                _key=_key.replace('sdata', 'dtype'),
                temp=temp)

    else:
        shared_arr[:] = arr[:]

    return _key


def shm_pop(key, keep=None):
    """Return key from shm.

    :param keep: Whether to keep the shm or erase it.
    If not specified, will delete shm if key ends with '_TEMP'
    (e.g. it was created with temp=True to shm_put)
    """
    result = SharedArray.attach(key)

    _, arrtype, _, tmp = key[6:].split('_')

    is_temp = tmp == 'TEMP'
    if keep is None:
        keep = not is_temp

    if arrtype == "sdata":
        # This was a structured array. Recover the dtype:
        dtype = unpack_dtype(shm_pop(key.replace('sdata', 'dtype'),
                                     keep=keep))
        result = np.frombuffer(result, dtype=dtype)

    elif arrtype == 'wrapper':
        # This was an array wrapper. Recover it, then its data
        result = unpickle_from_array(result)
        for x, fieldname in _array_wrappers.items():
            if isinstance(result, x):
                data = shm_pop(getattr(result, fieldname), keep=keep)
                setattr(result, fieldname, data)
                break
        else:
            raise RuntimeError(f"Popped weird thing from shm: {type(result)}")


    if not keep:
        shm_del(key)
    return result


def shm_del(key):
    return SharedArray.delete(key[6:])


def is_shmkey(key):
    return isinstance(key, str) and key.startswith('shm://npshmex')


def shm_wrap_f(f, *args, shm_out=True, **kwargs):
    # Get any shared memory args/kwargs
    for x in args:
        if is_shmable(x):
            raise ValueError(f"{f} got unshmd {type(x)} in args!")
    for k, x in kwargs.items():
        if is_shmable(x):
            raise ValueError(f"{f} got unshmd {type(x)} in kwargs {k}!")

    args = [shm_pop(x)
            if is_shmkey(x) else x
            for x in args]
    for k, v in kwargs.items():
        if is_shmkey(v):
            kwargs[k] = shm_pop(v)

    result = f(*args, **kwargs)

    if shm_out:
        if is_shmable(result):
            result = shm_put(result, temp=True)
        elif isinstance(result, dict):
            for k, v in result.items():
                if is_shmable(v):
                    result[k] = shm_put(v, temp=True)
    return result


def unshm(x, keep=None):
    """Retrieve x from shared memory if it is a shared memory key
    If x is a dict, replace any shared memory keys in values by numpy arrays.
    """
    if is_shmkey(x):
        x = shm_pop(x, keep=keep)
    elif isinstance(x, dict):
        for k, v in x.items():
            if is_shmkey(v):
                x[k] = shm_pop(v, keep=keep)
    return x


class ProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
    """ProcessPoolExecutor that passes numpy arrays in and out of the
    job functions through shared memory, avoiding the pickling overhead
    in python multiprocessing.
    """

    def submit(self, f, *args, shm_input=True, shm_output=True, **kwargs):
        """Return future for f(*args, **kwargs) computation.

        :param shm_input: If True (default), transfer numpy array input
        through shared memory
        :param shm_output: If True (default), transfer output of f via shared
        memory if it is a numpy array (or dict containing possibly numpy
        arrays).
        NB: result will contain un-attached shared memory keys! Use
        unshm to unpack the result.
        """
        if shm_input:
            # Copy numpy arguments to f to temporary shared memory
            args = [shm_put(x, temp=True) if is_shmable(x) else x
                    for x in args]
            for k, v in kwargs.items():
                if is_shmable(v):
                    kwargs[k] = shm_put(v, temp=True)

        if shm_input or shm_output:
            return super().submit(shm_wrap_f, f, *args,
                                  shm_out=shm_output, **kwargs)
        else:
            return super().submit(f, *args, **kwargs)

##
# Patch base_future to unpack results from shm as soon as they are put
# This is dirty, but I don't know another safe way
# (e.g. adding callback is not OK, someone can get at the result
#  before the callback runs)
##

class Future(concurrent.futures._base.Future):

    def set_result(self, result):
        super().set_result(unshm(result))

concurrent.futures._base.Future = Future
