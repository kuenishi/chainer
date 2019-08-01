import copy
import six

from chainer.training import extension
from chainer.iterators import SerialIterator
from chainer import backend
from chainer.dataset import convert
from chainer import function
from chainer import reporter as reporter_module


class MultiNodeAggregationEvaluator(extension.Extension):
    '''MultiNodeEvaluator for non-allreducable evaluation

    '''
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, comm, iterator, target, device=None,
                 eval_func=None):
        self.comm = comm
        self.iterator = iterator
        self._targets = {"main": target}
        self.eval_func = eval_func

        if device is not None:
            device = backend.get_device(device)
        self.device = device

    def initialize(self, trainer=None):
        self.iterator.reset()

    def __call__(self, trainer):
        # Set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        root = 0
        self.iterator.reset()
        g = self.evaluate_local(root)

        if self.comm.rank == root:
            self.aggregate(g)
        else:
            for _ in g:
                pass

    def evaluate_local(self, root):
        rounds = 8 #  Checks whether local eval is all done every 8 rounds
        all_done = None
        eval_func = self.eval_func or self._target['main']
        while not all_done:
            all_done = None
            results = None
            rest_values = None
            for i in range(rounds):
                try:
                    batch = self.iterator.next()
                    in_arrays, rest_values = self.preprocess(batch)

                    with function.no_backprop_mode():
                        if isinstance(in_arrays, tuple):
                            results = eval_func(*in_arrays)
                        elif isinstance(in_arrays, dict):
                            results = eval_func(**in_arrays)
                        else:
                            results = eval_func(in_arrays)

                    del batch
                    result = self.postprocess(in_arrays, results, rest_values)
                    del in_arrays

                except StopIteration:
                    result = None

                results = self.comm.gather_obj(result, root=root)

                if self.comm.rank == root:
                    valid_results = [r for r in results if r is not None]
                    for result in valid_results:
                        yield result

                    all_done = len(valid_results) == 0

            all_done = self.comm.bcast_obj(all_done, root=root)
        return

    def preprocess(self, batch):
        # batch is obtained from iterator.next()
        in_array = convert.concat_examples(batch)
        return in_array, None

    def postprocess(self, in_array, results, rest=None):
        # results obtained from eval_func or target model
        return results

    def aggregate(self, results_gen):
        # results_gen: generator object from postprocess at each
        # process
        raise NotImplementedError()

    def serialize(self, serializer):
        # TODO(kuenishi):
        raise NotImplementedError()


class GatherEvaluator(extension.Extension):
    '''MultiNodeEvaluator for non-allreducable evaluation

    '''
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, comm, iterator, target, aggregate_func, device=None,
                 converter=None, eval_func=None, root=0):
        '''
        iterator: test data iterator, (works with uneven iterators
        target or eval_func must be non-None
        aggregate_func: fun (Iterator) ->
        '''
        self.comm = comm
        self.iterator = iterator
        self._targets = {"main": target}
        self.eval_func = eval_func
        assert callable(aggregate_func)
        self.aggregate_func = aggregate_func
        self.converter = converter

        if device is not None:
            device = backend.get_device(device)
        self.device = device

        assert 0 <= root and root < self.comm.size
        self.root = root

    def initialize(self, trainer=None):
        self.iterator.reset()

    def __call__(self, trainer):
        # Set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        if hasattr(self.iterator, 'reset'):
            self.iterator.reset()
            it = self.iterator
        else:
            it = copy.copy(self.iterator)

        # Or obtain target from trainer
        eval_func = self.eval_func or self._targets['main']
        g = self.evaluate_local(eval_func, it)

        if self.comm.rank == self.root:
            self.aggregate_func(g)
        else:
            for _ in g:
                pass

    def evaluate_local(self, eval_func, iterator):
        rounds = 8 #  Checks whether local eval is all done every 8 rounds
        all_done = None

        while not all_done:
            all_done = None
            results = None
            rest_values = None
            for i in range(rounds):
                try:
                    batch = iterator.next()

                    if self.converter:
                        in_arrays = convert._call_converter(self.converter,
                                                            batch, self.device)
                    else:
                        in_arrays = batch

                    with function.no_backprop_mode():
                        if isinstance(in_arrays, tuple):
                            results = eval_func(*in_arrays)
                        elif isinstance(in_arrays, dict):
                            results = eval_func(**in_arrays)
                        else:
                            results = eval_func(in_arrays)

                except StopIteration:
                    results = None

                results = self.comm.gather_obj(results, root=self.root)

                if self.comm.rank == self.root:
                    valid_results = [r for r in results if r is not None]
                    for result in valid_results:
                        yield result

                    all_done = len(valid_results) == 0

            all_done = self.comm.bcast_obj(all_done, root=self.root)
        return



def create_multi_node_evaluator(actual_evaluator, communicator):
    """Create a multi node evaluator from a normal evaluator.

    Actually this method patches the evaluator to work in multi node
    environment. This method adds several hidden attributes starting
    with `_mn_` prefix.

    Args:
        actual_evaluator: evaluator to be patched
            (e.g., ``chainer.training.extensions.Evaluator``)
        communicator: ChainerMN communicator

    Returns:
        The multi-node patched ``actual_evaluator``.

    .. note:: After patched, original evaluator does not work
              correctly in non-MPI environment.

    """

    actual_evaluator._mn_original_evaluate = actual_evaluator.evaluate
    actual_evaluator._mn_communicator = communicator

    def new_evaluate(self):
        local_mean_dict = self._mn_original_evaluate()
        global_mean_dict = {
            name:
            self._mn_communicator.allreduce_obj(
                value) / self._mn_communicator.size
            for name, value in sorted(local_mean_dict.items())
        }
        return global_mean_dict

    actual_evaluator.evaluate = six.create_bound_method(
        new_evaluate, actual_evaluator)
    return actual_evaluator
