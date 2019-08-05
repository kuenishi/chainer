import collections
import os

import numpy as np

import chainer.cuda


class NullLatencyStats(object):
    def __init__(self, result, rank=0, verbose=False):
        pass

    def check_stream(self, stream):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def finalize(self):
        pass


# class CpuLatencyStats(object):
    
class GpuKernelLatencyStats(object):
    def __init__(self, result, rank=0, verbose=False):
        self.have_record = False
        self._start = chainer.cuda.Event(block=False)
        self._stop = chainer.cuda.Event(block=False)
        self.latencies = collections.deque(maxlen=1024)
        self.stream = chainer.cuda.Stream.null
        self.rank = rank
        
        self.result = result

        filename = os.path.join(result, "latency.{}.log".format(self.rank))
        self.out = open(filename, 'w')

        self.verbose = verbose

    def check_stream(self, stream):
        if self.stream != stream:
            raise ValueError('latency_stats must work with'
                             ' other stream than default.')

    def start(self):
        # To avoid sync latency penalty, latency is get right before
        # taking next measurement
        if self.have_record:
            self._stop.synchronize()
            t = chainer.cuda.cuda.get_elapsed_time(self._start,
                                                   self._stop)
            self.latencies.append(t)
            self.out.write(f"{t}\n")

        self._start.record(self.stream)
        self.have_record = False

    def stop(self):
        self._stop.record(self.stream)
        self.have_record = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def finalize(self):
        self.out.close()
        if self.verbose:
            self.print_stats()
        
    def print_stats(self):
        # Print stats
        mean = np.mean(self.latencies)
        print('allreduce letency stats (last 1024): rank', self.rank,
              'max', np.max(self.latencies),
              'min', np.min(self.latencies),
              'mean', mean,
              'stdev', sum( (x-mean) ** 2 for x in self.latencies ) / (len(self.latencies) - 1))

