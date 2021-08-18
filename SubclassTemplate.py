'''
SubclassTemplate.py
Written using Python 2.7.12 and TensorFlow 1.10
@ Matt Golub, June 2021.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from RecurrentWhisperer import RecurrentWhisperer

class SubclassTemplate(RecurrentWhisperer):
	''' Template for subclassing RecurrentWhisperer. Everything here is copied
	from RecurrentWhisperer.py, but includes only the minimal set of functions
	that must be implemented by any RecurrentWhisperer subclass.

	Technically, the following functions are not required in all use cases, but
	are included here for completeness:

    Required only if generating (or augmenting) data on-the-fly during
    training:
	    generate_data(...)

    Required only if do_batch_predictions:
	    _combine_prediction_batches(...)

    Not required, but can provide additional helpful functionality:
	    _setup_training(...)
	    _update_valid_tensorboard_summaries(...)
	    _update_visualizations(...)
	'''

    @staticmethod
    def _default_hash_hyperparameters():
        '''Defines subclass-specific default hyperparameters for the set of
        hyperparameters that are hashed to define a directory structure for
        easily managing multiple runs of the model training (i.e., using
        different hyperparameter settings). These hyperparameters may affect
        the model architecture or the trajectory of fitting, and as such are
        typically  swept during hyperparameter optimization.

        Values provided here will override the defaults set in this superclass
        for any hyperparameter keys that are overlapping with those in
        _default_super_hash_hyperparameters. Note, such overriding sets the
        default values for the subclass, and these subclass defaults can then
        be again overridden via keyword arguments input to __init__.

        Args:
            None.

        Returns:
            A dict of hyperparameters.
        '''
        return {}

    @staticmethod
    def _default_non_hash_hyperparameters():
        '''Defines default hyperparameters for the set of hyperparameters that
        are NOT hashed to define a run directory. These hyperparameters should
        not influence the model architecture or the trajectory of fitting.

        Values provided here will override the defaults set in this superclass
        for any hyperparameter keys that are overlapping with those in
        _default_super_non_hash_hyperparameters. Note, such overriding sets the
        default values for the subclass, and these subclass defaults can then
        be again overridden via keyword arguments input to __init__.

        Args:
            None.

        Returns:
            A dict of hyperparameters.
        '''
        return {}

    def _setup_training(self, train_data, valid_data=None):
        '''Performs any tasks that must be completed before entering the
        training loop in self.train().

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        super(SubclassTemplate, self)._setup_training(
        	train_data, valid_data=valid_data)

    def _setup_model(self):
        '''Defines the Tensorflow model including:
            -tf.placeholders for input data and prediction targets
            -a mapping from inputs to predictions
            -a scalar loss op named self.loss for comparing predictions to
            targets, regularization, etc.

        Args:
            None.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _build_data_feed_dict(self, batch_data, do_train_mode=True):
        ''' Build the feed dict that provides data to the model.

        Args:
            batch_data: dict containing the data needed to build the feed dict.

            do_train_mode: bool indicating whether these data will be used for
            running the model in "train mode" (True) or "predict mode" (False).
            Default: True.

        Returns:
            dict with (TF placeholder, feed value) as (key, value) pairs.

        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _get_pred_ops(self):
        ''' Get the dict of TF ops to be evaluated with each forward pass
        of the model. These are run by _predict_batch().

        Args:
            None.

        Returns:
            dict with (string label, TF ops) as (key, value) pairs.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    @classmethod
    def _get_batch_size(cls, batch_data):
        '''Returns the number of training examples in a batch of training data.

        Args:
            batch_data: dict containing one batch of training data.

        Returns:
            int specifying the number of examples in batch_data.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

	@classmethod
    def _subselect_batch(cls, data, batch_idx):
        ''' Subselect a batch of data given the batch indices.

        Args:
            data: dict containing the to-be-subselected data.

            batch_idx: array-like of trial indices.

        Returns:
            subselected_data: dict containing the subselected data.
        '''

        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def generate_data(self, train_or_valid_str='train'):
        ''' Optionally generate data on-the-fly (e.g., during training), rather
        than relying on fixed sets of training and validation data. This is
        only called by train(...) when called using train_data=None.

        Args:
            train_or_valid_str:

        Returns:
            data: dict containing the generated data.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _combine_prediction_batches(self, pred_list, summary_list, idx_list):
        ''' Combines predictions and summaries across multiple batches. This is
        required by _train_epoch(...) and predict(...), which first split data
        into multiple batches before sequentially calling _train_batch(...) or
        _predict_batch(...), respectively, on each data batch.

        Args:
            pred_list: list of prediction dicts, each generated by
            _predict_batch(...)

            summary_list: list of summary dicts, each generated by
            _predict_batch(...).

        Returns:
            pred: a single prediction dict containing the combined predictions
            from pred_list.

            summary: a single summary dict containing the combined summaries
            from summary_list.
        '''

        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _update_valid_tensorboard_summaries(self, valid_summary):
        '''Updates the Tensorboard summaries corresponding to the validation
        data. Only called if do_save_tensorboard_summaries.

        Args:
            valid_summary: dict returned by predict().

        Returns:
            None.
        '''
        pass

    def _update_visualizations(self,
        data, pred, train_or_valid_str, version):
        '''Updates visualizations in self.figs.

        Args:
            data: dict.

            pred: dict containing the result from predict(data).

            train_or_valid_str: either 'train' or 'valid', indicating whether
            data contains training data or validation data, respectively.

            version: string indicating the state of the model, which can be
            used to select which figures to generate. Options are: 'ltl',
            'lvl', 'seso', or 'final'.

        Returns:
            None.
        '''
        print('\tGenerating %s %s visualizations.' %
            (version.upper(), train_or_valid_str))

        ''' RecurrentWhisperer will save your figures in the run directory and
        will log them in Tensorboard (as desired per hyperparameter settings).
        To leverage this functionality, just create your figures like so:

            FIG_WIDTH = 6 # inches
            FIG_HEIGHT = 6 # inches

            fig = self._get_fig('your_figure_title',
                width=FIG_WIDTH,
                height=FIG_HEIGHT)

            # Now generate your visualization on fig

        This will create a new figure if one doesn't exist, or grab the figure
        if it was already created. This convention allows the same
        visualization plotted at various points throughout optimization to be
        placed on the same figure, to be saved to the same filename, and to be
        logged to the same Tensorboard image.
        '''