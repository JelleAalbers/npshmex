npshmex
=======

[![Build Status](https://travis-ci.org/JelleAalbers/npshmex.svg?branch=master)](https://travis-ci.org/JelleAalbers/npshmex)

Npshmex provides a drop-in replacement for concurrent.futures.ProcessPoolExecutor,
using shared memory provided by [SharedArray](https://gitlab.com/tenzing/shared-array) 
(rather than pickle) to transfer numpy arrays between processes.

Synopsis:
```python
import numpy as np
from npshmex import ProcessPoolExecutor

def add_one(x):
    return x + 1

ex = ProcessPoolExecutor()
big_data = np.ones(int(2e7))

f = ex.submit(add_one, big_data)
print(f.result()[0])           # 2.0
```
The last two lines take about ~290 ms on my laptop, but ~1250 ms using 
`concurrent.futures.ProcessPoolExecutor`: more than a factor four difference.
To run this example, npshmex claims only half as much memory as the standard library
`ProcessPoolExecutor`, based on the threshold at which I get a MemoryError.

For the trivial `add_one` task, multiprocessing overhead is dominant even when
spawning a single child process (a bare `add_one(big_data)` takes ~55 ms).
However, since part of the multiprocessing overhead is in the parent process, 
it will also bottleneck more complex tasks when they are scaled over enough processes.

How it works
--------------

Python multiprocessing uses pickle to serialize data for transfer between processes.
When passing around large numpy arrays, this can quickly become a bottleneck. 

Npshmex's ProcessPoolExecutor-replacement instead transfers input and output numpy arrays
using shared memory (`/dev/shm`). 
Dictionary outputs with numpy arrays as values are also supported.
Only the shared-memory `filenames' are actually transferred between processes.

Note that npshmex copies data from numpy arrays into shared memory
to transfer them. It doesn't copy it again on retrieval; it just creates the
numpy array with the shared memory backing it.
Still, if you are transferring the same array back and forth, 
this amounts to two unnecessary memory copies.
You can avoid these, and the use of npshmex, by managing the shared memory yourself:
```python
from concurrent.futures import ProcessPoolExecutor
import SharedArray

def add_one(shm_key):
    x = SharedArray.attach(shm_key)    
    x += 1

shm_key = 'shm://test'
ex = ProcessPoolExecutor()

big_data = SharedArray.create(shm_key, int(2e7))
big_data += 1

f = ex.submit(add_one, shm_key)
f.result()
SharedArray.delete(shm_key)
print(x[0])     # 2.0
```
The last four lines now only take ~130 ms on my laptop, which is over
twice as fast as npshmex. However, as you can see, it involves 
a more substantial rewrite of your code.

Npshmex also supports numpy structured arrays: it transfers the content as a `np.void` array, 
and the dtype (encoded to bytes) using a separate array.


Clearing shared memory
------------------------

Npshmex tells SharedArray to mark shared memory for deletion as soon as it has created
numpy arrays back from it. As explained in the 
[SharedArray](https://gitlab.com/tenzing/shared-array) documention, 
you'll keep the numpy array until you lose the last reference to it 
(as with regular python objects).

If your program exits while data is being transfered between processes, 
some shared files will remain in `/dev/shm`. You can manually clear all npshmex-associated
shared memory from all processes on the machine with `npshmex.shm_clear()`. 
Otherwise, it will be up to you, your operating system, or your system administrator
to clean up the mess...


Array wrappers
---------------
You can also use npshmex for objects that wrap numpy arrays.
For example:
```python
import numpy as np

class Wrapper:
    wrapped_array: np.ndarray
    other_field = 'whatever'
    ...

from npshmex import register_array_wrapper
register_array_wrapper(Wrapper, 'wrapped_array')
```
Whenever your functions take or return a Wrapper, the wrapped array will be passed through shared memory without pickling it first. 
The rest of the wrapper will be pickled.