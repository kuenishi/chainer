import six

from chainer.iterators import SerialIterator
from chainer import backend
from chainer.dataset import convert
from chainer import function
from chainer.training import extension
from chainer import reporter as reporter_module


class MultiNodeAggregationEvaluator(extension.Extension):
    '''MultiNodeEvaluator for non-allreducable evaluation

    It has 3 plugin points:
    - eval_func (optional)
    - postproc_func (or eval_hook?, optional)
    - aggregate (required)
    '''
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, comm, iterator, target,
                 device=None, gather_batch=False, eval_func=None,
                 postproc_func=None):
        self.comm = comm
        self.iterator = iterator
        self.target = target
        self.converter = convert.concat_examples
        self.gather_batch = gather_batch
        self.eval_func = eval_func
        self.postproc_func = postproc_func

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

        reporter.add_observer(prefix, self.target)
        reporter.add_observers(prefix,
                               self.target.namedlinks(skipself=True))

        root = 0
        rounds = 128 #  Checks whether local eval is all done every 8 rounds
        results = []
        all_done = False
        while not all_done:
            for result in self.evaluate_local(rounds):
                results0 = self.comm.gather_obj(result, root=root)
                # ProgressBar can be put here
                if self.comm.rank == root:
                    valid_results = [r for r in results0 if r is not None]
                    all_done = len(valid_results) == 0
                    results.extend(valid_results)

            print("><", self.comm.rank)
            all_done = self.comm.bcast_obj(all_done, root=root)

        if self.comm.rank == root:
            with reporter:
                report = self.aggregate(results)
            reporter_module.report(report, self.target)

    def postprocess_local(self, batch, results):
        '''Postprocess the result of inference locally

        You might need both ground truth from batch and inference
        results from model. Override this if you want to do further
        postprocessing locally, in parallel.

        The results should not be ``None``. (TODO(kuenishi): is there
        a need to support None?)

        '''
        if self.gather_batch:
            results = (batch, results)
        return results

    def evaluate_local(self, rounds):
        self.iterator.reset()
        eval_func = self.eval_func or self.target

        for i in range(rounds):
            try:
                batch = self.iterator.next()

                if callable(self.preprocess):
                    batch = self.preprocess(batch)
                #print(self.comm.rank, [len(in_value) for in_value in batch])
                #print(self.comm.rank, [[ v.shape for v in in_value]
                #                       for in_value in batch])
                
                #in_arrays = convert._call_converter(
                #    self.converter, batch, self.device)
                in_arrays = batch
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        results = eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        results = eval_func(**in_arrays)
                    else:
                        results = eval_func(in_arrays)
                yield self.postprocess_local(batch, results)
            except StopIteration:
                yield None
        return

    def aggregate(self, results):
        raise NotImplementedError()

    # TODO: def serialize(self, serializer): ...


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
