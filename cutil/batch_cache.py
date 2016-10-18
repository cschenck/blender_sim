#!/usr/bin/env python

import bisect
import multiprocessing
import numpy as np
import Queue
import random
import signal
import sys
import threading
import time

# Borrowed from http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"

class BatchCache:
    def __init__(self, max_size, lookup_func, batch_size, all_keys, random_seed=long(time.time()*100000), pre_fill=False):
        random.seed(random_seed)
        # Numpy array of the actual objects for fast access.
        self.vals = np.empty([0], dtype=np.dtype(object))
        self.keys = np.empty([0], dtype=np.dtype(object))
        # key -> index in vals array
        self.idxs = dict()
        self.data_size = np.empty([0])
        self.data_hits = np.empty([0], dtype=np.int)
        
        self.data_lock = threading.Lock()
        
        self.lookup_func = lookup_func
        self.all_keys = all_keys
        self.batch_size = batch_size
        self.max_size = max_size

        self.key_weights = {None:1.0}

        # original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        # self.pool = multiprocessing.Pool(1)
        # signal.signal(signal.SIGINT, original_sigint_handler)
        
        if not pre_fill:
            #Make the first sample.
            for i in range(self.batch_size):
                self.__random_load(force_append=True)
        
        bytes_per_batch = np.sum(self.data_size)
        if bytes_per_batch > self.max_size:
            raise Exception("%d bytes requred per batch, but max size is set to %d." % (bytes_per_batch, self.max_size))

        if pre_fill:
            # Fill the cache.
            while np.sum(self.data_size) < self.max_size:
                self.__random_load(force_append=True)

        # Start the queue monitor.
        self.keep_alive = True
        self.thread = threading.Thread(target=self.__memory_monitor, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def __del__(self):
        self.keep_alive = False
        self.thread.join()
        # self.pool.close()
        
    def __memory_monitor(self):
        while self.keep_alive:
            if np.sum(self.data_hits) > 0:
                self.__random_load()
            time.sleep(0.01)
            
    def __random_load(self, force_append=False):
        # First find some data that hasn't been loaded already.
        key = None
        while key is None or key in self.idxs:
            if hasattr(self.all_keys, '__call__'):
                key = self.all_keys()
            else:
                key = self.all_keys[random.randint(0, len(self.all_keys)-1)]
            weight = self.key_weights[key] if key in self.key_weights else self.key_weights[None]
            if random.random() > weight:
                key = None
        val, nbytes = self.lookup_func(key)
        # val, nbytes = self.__multiproc_map(self.lookup_func, [key])
        self.put(key, val, nbytes, force_append=force_append)

    def set_key_weight(self, key, weight):
        self.key_weights[key] = weight

    def put(self, key, val, nbytes, force_append=False):        
        # Check to see if we should append that data or replace existing data.
        self.data_lock.acquire()
        if np.sum(self.data_size) < self.max_size or force_append: # Append
            self.vals = np.concatenate((self.vals, np.array([None], dtype=np.dtype(object))))
            self.keys = np.concatenate((self.keys, np.array([None], dtype=np.dtype(object))))
            self.data_size = np.concatenate((self.data_size, np.array([0])))
            self.data_hits = np.concatenate((self.data_size, np.array([0], dtype=np.int)))
            i = self.vals.shape[0] - 1
        else: # Replace
            if np.sum(self.data_hits) == 0:
                sys.stderr.write("Something went horribly wrong. __random_load was called and the cache is full, but none of the elements have been hit!")
                sys.stderr.flush()
                self.data_lock.release()
                return
            i = weighted_choice(zip(range(len(self.vals)), 1.0*self.data_hits/np.sum(self.data_hits)))
            if self.keys[i] in self.idxs:
                del self.idxs[self.keys[i]]
        
        self.vals[i] = val
        self.keys[i] = key
        self.data_size[i] = nbytes
        self.data_hits[i] = 0
        self.idxs[key] = i
        
        self.data_lock.release()

    def __multiproc_map(self, func, args):
        res = None
        while True:
            try:
                if res is None:
                    res = self.pool.map_async(func, args)
                ret = res.get(0.1) # Without the timeout this blocking call ignores all signals.
                break
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                self.pool.terminate()
                raise
            except multiprocessing.TimeoutError:
                pass

        return ret
        
        
    def getBatch(self, key_weights={None:1.0}):
        self.data_lock.acquire()

        idxs = random.sample(range(len(self.vals)), self.batch_size)

        # Turns out this is super slow.
        # ws = [key_weights[x] if x in key_weights else key_weights[None] for x in self.keys]
        # for i in range(1, len(ws)):
        #     ws[i] += ws[i-1]
        # idxs = [-1]*self.batch_size
        # for i in range(len(idxs)):
        #     idx = -1
        #     while idx in idxs:
        #         idx = bisect.bisect(ws, random.random()*ws[-1])
        #     idxs[i] = idx
        
        self.data_hits[idxs] += 1
        vals = self.vals[idxs]
        keys = self.keys[idxs]
        self.data_lock.release()
        return (keys, vals)
        
        
def main():
    def lookup_func(k):
        ret = np.zeros((5,5))
        ret.fill(k)
        return ret
        
    cache = BatchCache(6400*10, lookup_func, 32, list(range(1024)), random_seed=2)
    for batch in range(32):
        ret = cache.getBatch()
        for i in range(ret[0].shape[0]):
            k = ret[0][i]
            v = ret[1][i]
            assert(v.shape[0] == 5)
            assert(v.shape[1] == 5)
            assert(len(v.shape) == 2)
            for i in range(5):
                for j in range(5):
                    assert v[i,j] == k, "v[%d,%d]=%s, k=%s" % (i, j, str(v[i,j]), str(k))
    
    print("Success.")
    
    
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
