'''
EpochResults.py
Written using Python 2.7.12
@ Matt Golub, July 2021.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class EpochResults(object):
    ''' Helper class for organizing model predictions and prediction summaries
    in a given training epoch.

    This is a simple retrieval system for model predictions and prediction
    summaries, intended to eliminate redundant computation wherever possible
    without burdening the programmer with needing to know what has already been
    computed elsewhere.

    A caller requests a result. If the result is found (i.e., it has already
    been computed), it is returned. Otherwise the result is computed, stored,
    and returned.
    '''
    def __init__(self,
        model=None, # A RecurrentWhisperer instance
        train_data=None,
        valid_data=None,
        do_batch=None, # default from RecurrentWhisperer.predict()
        is_final=False # default from RecurrentWhisperer.predict()
        ):

        self.model = model
        self._data = {'train': train_data, 'valid': valid_data}
        self._results = self._get_blank_results()

        self.do_batch = do_batch
        self.is_final = is_final

    def get(self, dataset, do_train_mode):
        ''' Requests a result. If the result is found (i.e., it has already
        been computed), it is returned. Otherwise the result is computed,
        stored internally for future use, and returned.

        Args:
            dataset: 'train' or 'valid', indicating which dataset the
            desired results should reference.

            do_train_mode: bool indicating whether the desired results should be
            computed using "train mode". See RecurrentWhisperer().

        Returns:
            predictions: See RecurrentWhisperer.predict()

            summary: See RecurrentWhisperer.predict()
        '''

        results = self._get_results_leaf(dataset, do_train_mode)

        if results['summary'] is None or results['predictions'] is None:
            data = self._data[dataset]
            pred, summary = self.model.predict(data,
                do_train_mode=do_train_mode,
                is_final=self.is_final,
                do_batch=self.do_batch)

            results['predictions'] = pred
            results['summary'] = summary

        return results['predictions'], results['summary']

    def set(self,
        dataset='train',
        do_train_mode=True,
        predictions=None,
        summary=None):
        ''' Store externally computed results for future use. This is primarily
        intended to store "train mode" results computed via
        RecurrentWhisperer._train_epoch(). Hence the defaults dataset='train'
        and do_train_mode=True.

        Args:
            dataset: 'train' or 'valid', indicating which dataset the
            desired results should reference.

            do_train_mode: bool indicating whether the desired results should be
            computed using "train mode". See RecurrentWhisperer().

        Returns:
            None
        '''
        results = self._get_results_leaf(dataset, do_train_mode)
        results['predictions'] = predictions
        results['summary'] = summary


    def exists(self, dataset, do_train_mode):
        ''' Returns True if the requested results already exist (i.e., they
        have been computed and can be retrieved).

        Args:
            dataset: 'train' or 'valid', indicating which dataset the
            desired results should reference.

            do_train_mode: bool indicating whether the desired results should be
            computed using "train mode". See RecurrentWhisperer().
        '''
        results = self._get_results_leaf(dataset, do_train_mode)
        if results['summary'] is None or results['predictions'] is None:
            return False
        else:
            return True

    def _get_results_leaf(self, dataset, do_train_mode):

        self._validate_args(dataset, do_train_mode)
        if do_train_mode:
            mode_str = 'train_mode'
        else:
            mode_str = 'predict_mode'

        return self._results[dataset][mode_str]

    @classmethod
    def _get_blank_results(cls):
        ''' Returns a 3-layered dict:
            Layer 1: indexed by 'train' or 'valid'
            Layer 2: indexed by 'train_mode' or 'predict_mode'
            Layer 3: indexed by 'summary' or 'predictions'
        '''
        def get_blank_dataset_results():

            def get_blank_leaf():
                return {'summary': None, 'predictions': None}

            return {
                'train_mode': get_blank_leaf(),
                'predict_mode': get_blank_leaf(),
                }

        return {
            'train': get_blank_dataset_results(),
            'valid': get_blank_dataset_results(),
        }

    @classmethod
    def _validate_args(cls, dataset, do_train_mode):

        assert dataset in ['train', 'valid'], \
            ('dataset must be \'train\' or \'valid\', '
                'but was %s' % str(dataset))

        assert isinstance(do_train_mode, bool), \
            ('do_train_mode must have type: bool, but has type: %s.' %
                str(type(do_train_mode)))