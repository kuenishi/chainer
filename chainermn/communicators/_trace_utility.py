from abc import ABCMeta
from abc import abstractmethod
import os
import socket
import time

import six
import numpy as np

import chainer.cuda


class LatencyTracer(six.with_metaclass(ABCMeta)):
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def start(self):
        raise NotImplementedError()

    @abstractmethod
    def stop(self):
        raise NotImplementedError()
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def finalize(self):
        pass


class NullLatencyTracer(LatencyTracer):
    def __init__(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class LatencyLogger(object):
    filename_template = 'latency.{}.log'

    def __init__(self, rank, directory):
        self.filename = os.path.join(result, self.filename_template.format(self.rank))
        self.out = open(filename, 'w')
        column = "{}|{}\n".format(socket.gethostname(), self.rank)
        self.out.write(column)

    def append(self, latency):
        # Supposed to be milliseconds
        self.out.write("{}\n".format(t))

    def close(self):
        self.out.close()
        
        
class CpuLatencyTracer(LatencyTracer):
    def __init__(self, result, rank=0):
        super(CpuLatencyTracer, self).__init__()
        self.rank = rank
        self.result = result
        self.b = None
        self.e = None
        self.logger = LatencyLogger(rank, result)

    def start(self):
        self.b = time.time()

    def stop(self):
        self.e = time.time()
        # Milliseconds
        self.logger.append( (e - b) / 1000 )

    def finalize(self):
        self.logger.close()

    
class GpuKernelLatencyTracer(LatencyTracer):
    def __init__(self, result, rank=0):
        super(GpuKernelLatencyTracer, self).__init__()
        self.have_record = False
        self._start = chainer.cuda.Event(block=False)
        self._stop = chainer.cuda.Event(block=False)
        self.stream = chainer.cuda.Stream.null
        self.rank = rank
        self.result = result

        self.logger = LatencyLogger(rank, result)

    def start(self):
        # To avoid sync latency penalty, latency is get right before
        # taking next measurement
        if self.have_record:
            self._stop.synchronize()
            t = chainer.cuda.cuda.get_elapsed_time(self._start,
                                                   self._stop)
            self.logger.append(t)

        self._start.record(self.stream)
        self.have_record = False

    def stop(self):
        self._stop.record(self.stream)
        self.have_record = True

    def finalize(self):
        self.logger.close()
