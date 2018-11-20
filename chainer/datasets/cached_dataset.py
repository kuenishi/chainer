import os
import pickle
import tempfile

import chainer
from chainer.dataset import dataset_mixin

class CachedDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataset, postprocess=None,
                 cache_after_postprocess=False,
                 serializer=(pickle.loads, pickle.dumps),
                 path='/tmp'):
        self.dataset = dataset
        self.postprocess = postprocess
        self.cache_after_postprocess = cache_after_postprocess
        self.deserializer = serializer[0]
        self.serializer = serializer[1]

        # TODO: use temporary file instead
        # TODO: this doesn't work with MultiprocessIterator, possibly?
        #self.cache_file = open(path, 'wb')
        (handle, name) = tempfile.mkstemp(dir=path)
        self.cache_file = handle
        self.filename = name
        # TODO: use array instead
        self.offsets = {}
        self.pos = 0
        if not self.cache_file:
            raise ValueError("Cannot open file for cache")

    def __del__(self):
        os.close(self.cache_file)

    def __len__(self):
        return len(self.dataset)

    def _do_cache(self, example, i):
        buf = self.serializer(example)
        size = os.pwrite(self.cache_file, buf, self.pos)
        # TODO: think of which exceptions to throw
        assert size > 0
        self.offsets[i] = (self.pos, size)
        self.pos += size
        
    def get_example(self, i):
        if i in self.offsets:
            (offset, length) = self.offsets[i]
            example = self.deserializer(os.pread(self.cache_file, length, offset))
            if not self.cache_after_postprocess and self.postprocess is not None:
                    example = self.postprocess(example)
            return example

        # print('no cache hit:', i)
        example = self.dataset[i]
        if self.cache_after_postprocess:
            if self.postprocess is not None:
                example = self.postprocess(example)
            self._do_cache(example, i)
        else:
            self._do_cache(example, i)
            if self.postprocess is not None:
                example = self.postprocess(example)
        return example
                
            
        
        
        
