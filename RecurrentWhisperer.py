'''
RecurrentWhisperer.py
Version 1.0
Written using Python 2.7.12 and TensorFlow 1.10
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
import ntpath
from copy import deepcopy
import pdb

import tensorflow as tf
import numpy as np
import numpy.random as npr
import cPickle
import scipy.io as spio

from AdaptiveLearningRate import AdaptiveLearningRate
from AdaptiveGradNormClip import AdaptiveGradNormClip
from Hyperparameters import Hyperparameters
from timer import timer


class RecurrentWhisperer(object):
    '''A general class template for training recurrent neural networks using
    TensorFlow. This class provides functionality for:

    1) Training a recurrent neural network using modern techniques for
    encouraging stable training, such as adaptive learning rates and adaptive
    gradient norm clipping. This class handles common tasks like splitting
    training data into batches, making gradient steps based on individual
    batches of training data, periodically evaluating validation data, and
    periodically saving model checkpoints.

    2) Managing Tensorboard visualizations of training progress.

    3) Managing a directory structure for maintaining many different variants
    of a model (i.e., with different hyperparameter settings). Previously
    saved models can be readily restored from checkpoints, and training runs
    can be readily resumed if their execution was interrupted or preempted.

    Subclasses inheriting from RecurrentWhisperer must implement the following
    functions (see definitions in this file for call signatures):

    _default_hash_hyperparameters()
    _default_non_hash_hyperparameters()
    _setup_model(...)
    _setup_training(...)
    _get_data_batches(...)
    _get_batch_size(...)
    _train_batch(...)
    predict(...)
    _setup_visualization(...)
    _update_visualization(...)
    '''

    _DEFAULT_SUPER_HASH_HPS = {
        'random_seed': 0,
        'dtype': 'float32', # keep as string (rather than tf.float32)
                            # for better argparse handling, yaml writing
        'adam_hps': {'epsilon': 0.01},
        'alr_hps': {},
        'agnc_hps': {}}

    _DEFAULT_SUPER_NON_HASH_HPS = {
        'min_loss': 0.,
        'max_n_epochs': 1000,
        'max_n_epochs_without_lvl_improvement': 200,
        'min_learning_rate': 1e-10,
        'log_dir': '/tmp/rnn_logs/',
        'do_save_lvl_mat_files': False,
        'do_restart_run': False,
        'max_ckpt_to_keep': 1,
        'max_lvl_ckpt_to_keep': 1,
        'n_epochs_per_ckpt': 100,
        'n_epochs_per_validation_update': 100,
        'n_epochs_per_visualization_update': 100,
        'disable_gpus': False,
        'allow_gpu_growth': True,
        'per_process_gpu_memory_fraction': None}

    def __init__(self, **kwargs):
        '''Creates a RecurrentWhisperer object.

        Args:
            A set of optional keyword arguments for overriding default
            hyperparameter values. Hyperparameters are grouped into 2
            categories--those that affect the trajectory of training (e.g.,
            learning rate), and those that do not (e.g., logging preferences).
            Those in the former category are hashed to yield a unique run
            directory for saving checkpoints, Tensorboard events, etc. Those
            in the latter category are included in this hash so that one can
            more readily interact with a training run without requiring a
            complete restart (e.g., change printing or visualization
            preferences; change optimization termination criteria).

                See also:
                    _default_hash_hyperparameters
                    _default_non_hash_hyperparameters

            Hyperparameters included in the run directory hash (defined in
            _default_hash_hyperparameters):

                random_seed: int specifying the random seed for the numpy
                random generator used for randomly batching data and
                initializing model parameters. Default: 0

                dtype: string indicating the Tensorflow data type to use for
                all Tensorflow objects. Default: 'float32' --> tf.float32.

                adam_hps: dict specifying hyperparameters for TF's
                AdamOptimizer. Default: {'epsilon': 0.01}. See
                tf.AdamOptimizer.

                alr_hps: dict specifying hyperparameters for managing an
                adaptive learning rate. Default: set by AdaptiveLearningRate.

                agnc_hps: dict specifying hyperparameters for managing
                adaptive gradient norm clipping. Default: set by
                AdaptiveGradNormClip.

            Hyperparameters not included in the run directory hash (defined in
            _default_non_hash_hyperparamet:

                min_loss: float specifying optimization termination criteria
                on the loss function evaluated across the training data (the
                epoch training loss). Default: 0.

                max_n_epochs: int specifying optimization termination criteria
                on the number of training epochs performed (one epoch = one
                full pass through the entire training dataset). Default: 1000.

                max_n_epochs_without_lvl_improvement: int specifying
                optimization termination criteria on the number of training
                epochs performed without improvements to the lowest validation
                loss. If the lowest validation error does not improve over a
                block of this many epochs, training will terminate. If
                validation data are not provided to train(...), this
                termination criteria is ignored. Default: 200.

                min_learning_rate: float specifying optimization termination
                criteria on the learning rate. Optimization terminates if the
                learning rate falls below this value. Default: 1e-10.

                log_dir: string specifying the top-level directory for saving
                various training runs (where each training run is specified by
                a different set of hyperparameter settings). When tuning
                hyperparameters, log_dir is meant to be constant across
                models. Default: '/tmp/rnn_logs/'.

                do_save_lvl_mat_files: bool indicating whether to save .mat
                files containing predictions over the training and validation
                data each time a new lowest validation loss is achieved.
                Regardless of this setting, .pkl files are saved. Default:
                False.

                do_restart_run: bool indicating whether to force a restart of
                a training run (e.g., if a previous run with the same
                hyperparameters has saved checkpoints--the previous run will
                be deleted and restarted rather than resumed). Default: False.

                max_ckpt_to_keep: int specifying the maximum number of model
                checkpoints to keep around. Default: 1.

                max_lvl_ckpt_to_keep: int specifying the maximum number
                of lowest validation loss (lvl) checkpoints to maintain.
                Default: 1.

                n_epochs_per_ckpt: int specifying the number of epochs between
                checkpoint saves. Default: 100.

                n_epochs_per_validation_update: int specifying the number of
                epochs between evaluating predictions over the validation
                data. Default: 100.

                n_epochs_per_visualization_update: int specifying the number
                of epochs between updates of any visualizations. Default: 100.

                disable_gpus: bool indicating whether to disable access to any
                GPUs. Default: False.

                allow_gpu_growth: bool indicating whether to dynamically
                allocate GPU memory (True) or to monopolize the entire memory
                capacity of a GPU (False). Default: True.

                per_process_gpu_memory_fraction: float specifying the maximum
                fraction of GPU memory to allocate. Set to None to allow
                Tensorflow to manage GPU memory. See Tensorflow documentation
                for interactions between device_count (accessed here via
                disable_gpus), enable_gpu_growth, and
                per_process_gpu_memory_fraction. Default: None.

        Returns:
            None.
        '''
        default_hash_hps = self._integrate_hps(
            self._DEFAULT_SUPER_HASH_HPS,
            self._default_hash_hyperparameters())

        default_non_hash_hps = self._integrate_hps(
            self._DEFAULT_SUPER_NON_HASH_HPS,
            self._default_non_hash_hyperparameters())

        hps = Hyperparameters(kwargs, default_hash_hps, default_non_hash_hps)
        self.hps = hps
        self.dtype = getattr(tf, hps.dtype)

        '''Make parameter initializations and data batching reproducible
        across runs. Because this gets updated as data are drawn from the
        random number generator, currently this state will not transfer across
        saves and restores.'''
        self.rng = npr.RandomState(hps.random_seed)

        self.adaptive_learning_rate = AdaptiveLearningRate(**hps.alr_hps)
        self.adaptive_grad_norm_clip = AdaptiveGradNormClip(**hps.agnc_hps)

        ckpt = self._setup_run_dir()
        self._setup_model()
        self._setup_optimizer()
        self._setup_visualization()

        # Each of these will create run_dir if it doesn't exist
        # (do not move above the os.path.isdir check that is in _setup_run_dir)
        self._setup_tensorboard()
        self._setup_savers()

        self._setup_session()
        self._initialize_or_restore()

    @staticmethod
    def _integrate_hps(superclass_default_hps, subclass_default_hps):
        '''Integrates default hyperparameters defined in this superclass with
        those defined in a subclass. Subclasses may override defaults
        specified by this superclass.

        Args:
            superclass_default_hps: dict containing default hyperparameters
            required by this superclass.

            subclass_default_hps: dict containing default hyperparameters that
            are subclass specific or override defaults set by this superclass.

        Returns:
            default_hps: dict containing the integrated hyperparameters.
        '''
        default_hps = deepcopy(superclass_default_hps)
        for key, val in subclass_default_hps.iteritems():
            default_hps[key] = val

        return default_hps

    @staticmethod
    def get_paths(log_dir, run_hash):
        '''Generates all paths relevant for saving and loading model data.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

        Returns:
            dict containing all paths relevant for saving and loading model
            data. Keys are strings, with suffixes '_dir' and '_path' referring
            to directories and filenames, respectively.
        '''

        run_dir = os.path.join(log_dir, run_hash)
        hps_dir = os.path.join(run_dir, 'hps')
        ckpt_dir = os.path.join(run_dir, 'ckpt')
        lvl_dir = os.path.join(run_dir, 'lvl')

        return {
            'run_dir': run_dir,
            'hps_dir': hps_dir,
            'hps_path': os.path.join(hps_dir, 'hyperparameters.pkl'),
            'hps_yaml_path': os.path.join(hps_dir, 'hyperparameters.yml'),
            'alr_path': os.path.join(hps_dir, 'learn_rate.pkl'),
            'agnc_path': os.path.join(hps_dir, 'norm_clip.pkl'),
            'ckpt_dir': ckpt_dir,
            'ckpt_path': os.path.join(ckpt_dir, 'checkpoint.ckpt'),
            'lvl_dir': lvl_dir,
            'lvl_ckpt_path': os.path.join(lvl_dir, 'lvl.ckpt'),
            'events_dir': os.path.join(run_dir, 'events')
            }

    def _setup_run_dir(self):
        '''Sets up a directory for this training run. The directory name is
        derived from a hash of the hyperparameter settings. Subdirectories are
        also managed for saving/restoring hyperparameters, model checkpoints,
        and Tensorboard events.

        Args:
            None.

        Returns:
            None.
        '''
        hps = self.hps
        log_dir = hps.log_dir
        run_hash = hps.get_hash()
        paths = self.get_paths(log_dir, run_hash)

        self.run_dir = paths['run_dir']
        self.hps_dir = paths['hps_dir']
        self.hps_path = paths['hps_path']
        self.hps_yaml_path = paths['hps_yaml_path']
        self.alr_path = paths['alr_path']
        self.agnc_path = paths['agnc_path']

        self.ckpt_dir = paths['ckpt_dir']
        self.ckpt_path = paths['ckpt_path']

        self.lvl_dir = paths['lvl_dir']
        self.lvl_ckpt_path = paths['lvl_ckpt_path']

        # For managing Tensorboard events
        self.events_dir = paths['events_dir']

        if os.path.isdir(self.run_dir):
            print('\nRun directory found: %s.' % self.run_dir)
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            if ckpt is None:
                print('No checkpoints found.')
            if self.hps.do_restart_run:
                print('\tDeleting run directory.')
                shutil.rmtree(self.run_dir)

        if not os.path.isdir(self.run_dir):
            print('\nCreating run directory: %s.' % self.run_dir)
            os.makedirs(self.run_dir)
            os.makedirs(self.hps_dir)
            os.makedirs(self.ckpt_dir)
            os.makedirs(self.lvl_dir)
            os.makedirs(self.events_dir)

    def _setup_optimizer(self):
        '''Sets up an AdamOptimizer with gradient norm clipping, along with
        supporting variables (and corresponding Tensorboard summaries) for
        tracking optimization progress.

        Args:
            None.

        Returns:
            None.
        '''
        with tf.name_scope('optimizer'):
            '''Maintain state using TF framework for seamless saving and
            restoring of runs'''
            self.global_step = tf.Variable(
                0, name='global_step', trainable=False, dtype=tf.int32)

            self.epoch = tf.Variable(
                0, name='epoch', trainable=False, dtype=tf.int32)

            self.increment_epoch = tf.assign_add(
                self.epoch, 1, name='increment_epoch')

            # lowest validation loss
            self.lvl = tf.Variable(
                np.inf, name='lvl', trainable=False, dtype=self.dtype)

            self.epoch_last_lvl_improvement = tf.Variable(
                0, name='epoch_last_lvl_improvement',
                trainable=False, dtype=self.dtype)

            # lowest training loss
            self.ltl = tf.Variable(
                np.inf, name='ltl', trainable=False, dtype=self.dtype)

            # Gradient clipping
            grads = tf.gradients(self.loss, tf.trainable_variables())

            self.grad_norm_clip_val = tf.placeholder(
                self.dtype, name='grad_norm_clip_val')

            clipped_grads, self.grad_global_norm = tf.clip_by_global_norm(
                grads, self.grad_norm_clip_val)

            clipped_grad_global_norm = tf.global_norm(clipped_grads)

            clipped_grad_norm_diff = \
                self.grad_global_norm - clipped_grad_global_norm

            zipped_grads = zip(clipped_grads, tf.trainable_variables())

            self.learning_rate = tf.placeholder(
                self.dtype, name='learning_rate')

            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, **self.hps.adam_hps)

            self.train_op = optimizer.apply_gradients(
                zipped_grads, global_step=self.global_step)

        # Tensorboard summaries (updated during training via _train_epoch)
        with tf.name_scope('tb-optimizer'):
            summaries = []
            summaries.append(tf.summary.scalar('loss', self.loss))
            summaries.append(tf.summary.scalar('lvl', self.lvl))
            summaries.append(tf.summary.scalar('learning rate',
                                               self.learning_rate))
            summaries.append(tf.summary.scalar('grad global norm',
                                               self.grad_global_norm))
            summaries.append(tf.summary.scalar('grad norm clip val',
                                               self.grad_norm_clip_val))
            summaries.append(tf.summary.scalar('clipped grad global norm',
                                               clipped_grad_global_norm))
            summaries.append(tf.summary.scalar('grad clip diff',
                                               clipped_grad_norm_diff))

            self.merged_opt_summary = tf.summary.merge(summaries)

    def _setup_tensorboard(self):
        '''Sets up the Tensorboard FileWriter.

        Args:
            None.

        Returns:
            None.
        '''
        self.writer = tf.summary.FileWriter(self.events_dir)
        self.writer.add_graph(tf.get_default_graph())

    def _setup_savers(self):
        '''Sets up Tensorflow checkpoint saving.

        Args:
            None.

        Returns:
            None.
        '''
        # save every so often
        self.seso_saver = tf.train.Saver(tf.global_variables(),
                                         max_to_keep=self.hps.max_ckpt_to_keep)
        # lowest validation loss
        self.lvl_saver = \
            tf.train.Saver(tf.global_variables(),
                           max_to_keep=self.hps.max_lvl_ckpt_to_keep)

    def _setup_session(self):
        '''Sets up a Tensorflow session with the desired GPU configuration.

        Args:
            None.

        Returns:
            None.
        '''
        hps = self.hps

        if hps.disable_gpus:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto()

        config.gpu_options.allow_growth = hps.allow_gpu_growth
        if hps.per_process_gpu_memory_fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = \
                hps.per_process_gpu_memory_fraction

        self.session = tf.Session(config=config)
        print('\n')

    def _initialize_or_restore(self):
        '''Initializes all Tensorflow objects, either from existing model
        checkpoint if detected or otherwise as specified in _setup_model. If
        starting a training run from scratch, writes a yaml file containing
        all hyperparameter settings.

        Args:
            None.

        Returns:
            None.
        '''
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)

        if ckpt is None:
            # Initialize new session
            print('Initializing new run (%s).' % self.hps.get_hash())
            self.session.run(tf.global_variables_initializer())

            self.hps.write_yaml(self.hps_yaml_path) # For visual inspection
            self.hps.save(self.hps_path) # For restoring a run via its run_dir
            # (i.e., without needing to manually specify hps)
        else:
            self._restore_from_checkpoint(ckpt)

    def train(self, train_data=None, valid_data=None):
        '''Trains the model with respect to a set of training data. Manages
        the core tasks involved in model training:
            -randomly batching the training data
            -updating model parameters via gradients over each data batch
            -periodically evaluating the validation data
            -periodically updating visualization
            -periodically saving model checkpoints

        By convention, this call is the first time this class object has
        access to the data (train and valid).

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''

        N_TIMER_SPLITS = 6

        self._setup_training(train_data, valid_data)

        if self._is_training_complete(self._ltl()):
            # If restoring from a completed run, do not enter training loop
            # and do not save a new checkpoint.
            return

        # Training loop
        print('Entering training loop.')
        while True:
            t = timer(N_TIMER_SPLITS, n_indent=1)
            t.start()

            data_batches = self._get_data_batches(train_data)

            t.split('data')

            epoch_loss = self._train_epoch(data_batches)

            t.split('train')

            self._maybe_update_validation(train_data, valid_data)

            t.split('validation')

            self._maybe_update_visualization(train_data, valid_data)

            t.split('visualize')

            self._maybe_save_checkpoint()

            t.split('save')

            if self._is_training_complete(epoch_loss, valid_data is not None):
                break

            t.split('other')
            t.disp()

        # Save checkpoint upon completing training
        self._save_checkpoint()

    def _is_training_complete(self, loss, do_check_lvl=False):
        '''Determines whether the training optimization procedure should
        terminate. Termination criteria, goverened by hyperparameters, are
        thresholds on the following:

            1) the training loss
            2) the learning rate
            3) the number of training epochs performed
            4) the number of training epochs performed since the lowest
               validation loss improved (only if validation data are provided
               to train(...).

        Args:
            loss: float indicating the most recent loss evaluation across the
            training data.

        Returns:
            bool indicating whether any of the termination criteria have been
            met.
        '''
        hps = self.hps

        if loss <= hps.min_loss:
            print ('\nStopping optimization: loss meets convergence criteria.')
            return True

        if self._epoch() > self.adaptive_learning_rate.n_warmup_steps and \
            self.adaptive_learning_rate() <= hps.min_learning_rate:
            print ('\nStopping optimization: minimum learning rate reached.')
            return True

        if self._epoch() >= hps.max_n_epochs:
            print('\nStopping optimization:'
                  ' reached maximum number of training epochs.')
            return True

        if do_check_lvl and \
            self._epoch() - self._epoch_last_lvl_improvement() >= \
                hps.max_n_epochs_without_lvl_improvement:

            print('\nStopping optimization:'
                  ' reached maximum number of training epochs'
                  ' without improvement to the lowest validation loss.')
            return True

        return False

    def _train_epoch(self, data_batches):
        '''Performs training steps across an epoch of training data batches.

        Args:
            data_batches: list of dicts, where each dict contains a batch of
            training data.

        Returns:
            epoch_loss: float indicating the average loss across the epoch of
            training batches.
        '''
        def compute_epoch_average(vals, n):
            '''Computes a weighted average of evaluations of a summary
            statistic across an epoch of data batches.

            Args:
                vals: list of floats containing the summary statistic
                evaluations to be averaged.

                n: list of ints containing the number of data examples per
                data batch.

            Returns:
                avg: float indicating the weighted average.
            '''
            weights = np.true_divide(n, np.sum(n))
            avg = np.dot(weights, vals)
            return avg

        n_batches = len(data_batches)

        batch_losses = np.zeros(n_batches)
        batch_grad_norms = np.zeros(n_batches)
        batch_sizes = np.zeros(n_batches)

        for batch_idx in range(n_batches):
            batch_data = data_batches[batch_idx]
            batch_summary = self._train_batch(batch_data)

            batch_losses[batch_idx] = batch_summary['loss']
            batch_grad_norms[batch_idx] = batch_summary['grad_global_norm']
            batch_sizes[batch_idx] = self._get_batch_size(batch_data)

        epoch_loss = compute_epoch_average(batch_losses, batch_sizes)

        # Update lowest training loss (if applicable)
        if epoch_loss < self._ltl():
            self.session.run(tf.assign(self.ltl, epoch_loss))

        self.adaptive_learning_rate.update(epoch_loss)

        epoch_grad_norm = compute_epoch_average(batch_grad_norms, batch_sizes)
        self.adaptive_grad_norm_clip.update(epoch_grad_norm)

        self._increment_epoch()

        print('Epoch %d; loss: %.2e; learning rate: %.2e.'
            % (self._epoch(), epoch_loss, self.adaptive_learning_rate()))

        return epoch_loss

    def _maybe_update_validation(self, train_data, valid_data):
        '''Evaluates the validation data if the current epoch number indicates
        that such an update is due.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''

        if valid_data is None:
            pass
        else:
            n = self.hps.n_epochs_per_validation_update
            if np.mod(self._epoch(), n) == 0:
                self._update_validation(train_data, valid_data)

    def _update_validation(self, train_data, valid_data):
        '''Evaluates the validation data, updates the corresponding
        Tensorboard summaries are updated, and if the validation loss
        indicates a new minimum, a model checkpoint is saved.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        predictions, summary = self.predict(valid_data)
        print('\tValidation loss: %.2e' % summary['loss'])

        self._update_valid_tensorboard(summary)
        self._maybe_save_lvl_checkpoint(
            summary['loss'], train_data, valid_data)

    def _maybe_update_visualization(self, train_data, valid_data):
        '''Updates visualizations if the current epoch number indicates that
        an update is due.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''

        n = self.hps.n_epochs_per_visualization_update
        if np.mod(self._epoch(), n) == 0:
            self._update_visualization(train_data, valid_data)

    def _maybe_save_checkpoint(self):
        '''Saves a model checkpoint if the current epoch number indicates that
        a checkpoint is due.

        Args:
            None.

        Returns:
            None.
        '''
        if np.mod(self._epoch(), self.hps.n_epochs_per_ckpt) == 0:
            self._save_checkpoint()

    def _save_checkpoint(self):
        '''Saves a model checkpoint.

        Args:
            None.

        Returns:
            None.
        '''
        print('\tSaving checkpoint...')
        self.seso_saver.save(self.session,
                             self.ckpt_path,
                             global_step=self._step())
        self.adaptive_learning_rate.save(self.alr_path)
        self.adaptive_grad_norm_clip.save(self.agnc_path)

    def _restore_from_checkpoint(self, ckpt):
        '''Restores a model from the most advanced previously saved model
        checkpoint.

        Args:
            None.

        Returns:
            None.
        '''
        if not(tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
            raise FileNotFoundError('Checkpoint does not exist: %s'
                                    % ckpt.model_checkpoint_path)

        # Restore previous session
        print('Previous checkpoints found.')
        print('Loading latest checkpoint: %s.'
              % ntpath.basename(ckpt.model_checkpoint_path))
        self.seso_saver.restore(self.session, ckpt.model_checkpoint_path)
        self.adaptive_learning_rate.restore(self.alr_path)
        self.adaptive_grad_norm_clip.restore(self.agnc_path)

    def _maybe_save_lvl_checkpoint(self, valid_loss, train_data, valid_data):
        '''Saves a model checkpoint if the current validation loss values is
        lower than all previously evaluated validation loss values.

        Args:
            valid_loss: float indicating the current loss evaluated over the
            validation data.

            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        if self._epoch()==0 or valid_loss < self._lvl():
            print('Achieved lowest validation loss. Saving checkpoint...')

            assign_ops = [tf.assign(self.lvl, valid_loss),
                tf.assign(self.epoch_last_lvl_improvement, self._epoch())]

            self.session.run(assign_ops)

            self.lvl_saver.save(self.session,
                                self.lvl_ckpt_path,
                                global_step=self._step())

            self.save_lvl_predictions(train_data, 'train')
            self.save_lvl_predictions(valid_data, 'valid')

    def restore_from_lvl_checkpoint(self):
        '''Restores a model from a previously saved lowest-validation-loss
        checkpoint.

        Args:
            None.

        Returns:
            None.

        Raises:
            FileNotFoundError (if no lowest-validation-loss checkpoint exists).
        '''
        lvl_ckpt = tf.train.get_checkpoint_state(self.lvl_dir)
        if not(tf.train.checkpoint_exists(lvl_ckpt.model_checkpoint_path)):
            raise FileNotFoundError('Checkpoint does not exist: %s'
                                    % lvl_ckpt.model_checkpoint_path)

        # Restore previous session
        print('\tLoading lvl checkpoint: %s.'
              % ntpath.basename(lvl_ckpt.model_checkpoint_path))
        self.lvl_saver.restore(self.session, lvl_ckpt.model_checkpoint_path)

    def save_lvl_predictions(self, data, train_or_valid_str):
        '''Saves all model predictions in .pkl files (and optionally in .mat
        files as well).

        If prediction summaries are generated, those summaries are saved in
        separate .pkl files (and optional .mat files). See docstring for
        predict() for additional detail.

        Args:
            data: dict containing either training or validation data.

            train_or_valid_str: either 'train' or 'valid', indicating wheter
            data contains training data or validation data, respectively.

        Returns:
            None.
        '''

        def save_helper(data_to_save, suffix_str):
            # E.g., train_predictions or valid_summary
            filename_no_extension = train_or_valid_str + '_' + suffix_str

            pkl_path = os.path.join(self.lvl_dir, filename_no_extension)
            save_pkl(data_to_save, pkl_path)

            if self.hps.do_save_lvl_mat_files:
                mat_path = os.path.join(self.lvl_dir, filename_no_extension)
                save_mat(data_to_save, pkl_path)

        def save_pkl(data_to_save, save_path_no_extension):
            '''Pickle and save data as .pkl file.

            Args:
                save_path_no_extension: path at which to save the data,
                including an extensionless filename.

                data_to_save: any pickle-able object to be pickled and saved.

            Returns:
                None.
            '''

            save_path = save_path_no_extension + '.pkl'
            file = open(save_path, 'wb')
            file.write(cPickle.dumps(data_to_save))
            file.close()

        def save_mat(data_to_save, save_path_no_extension):
            '''Save data as .mat file.

            Args:
                save_path_no_extension: path at which to save the data,
                including an extensionless filename.

                data_to_save: dict containing data to be saved.

            Returns:
                None.
            '''

            save_path = save_path_no_extension + '.mat'
            spio.savemat(save_path, data_to_save)

        print('\tSaving lvl predictions (%s).' % train_or_valid_str)
        predictions, summary = self.predict(data)
        save_helper(predictions, 'predictions')
        if summary is not None:
            save_helper(summary, 'summary')

    @staticmethod
    def _load_lvl_helper(log_dir, run_hash,
        train_or_valid_str, predictions_or_summary_str):
        '''Loads previously saved model predictions or summaries thereof.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

            train_or_valid_str: either 'train' or 'valid', indicating wheter
            to load predictions/summary from the training data or validation
            data, respectively.

            predictions_or_summary_str: either 'predictions' or 'summary',
            indicating whether to load model predictions or summaries thereof,
            respectively.

        Returns:
            dict containing saved data.
        '''
        paths = RecurrentWhisperer.get_paths(log_dir, run_hash)
        filename = train_or_valid_str + '_' + \
            predictions_or_summary_str + '.pkl'
        load_path = os.path.join(paths['lvl_dir'], filename)

        file = open(load_path, 'rb')
        load_path = file.read()
        file.close()
        return cPickle.loads(load_path)

    @staticmethod
    def load_lvl_train_predictions(log_dir, run_hash):
        '''Loads all model predictions made over the training data by the lvl
        model.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

        Returns:
            dict containing saved predictions.
        '''
        return RecurrentWhisperer._load_lvl_helper(
            log_dir, run_hash, 'train', 'predictions')

    @staticmethod
    def load_lvl_train_summary(log_dir, run_hash):
        '''Loads summary of the model predictions made over the training data
        by the lvl model.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

        Returns:
            dict containing saved summaries.
        '''
        return RecurrentWhisperer._load_lvl_helper(
            log_dir, run_hash, 'train', 'summary')

    @staticmethod
    def load_lvl_valid_predictions(log_dir, run_hash):
        '''Loads all model predictions from train_predictions.pkl.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

        Returns:
            train_predictions:
                dict containing saved predictions on the training data by the
                lvl model.
        '''
        return RecurrentWhisperer._load_lvl_helper(
            log_dir, run_hash, 'valid', 'predictions')

    @staticmethod
    def load_lvl_valid_summary(log_dir, run_hash):
        '''Loads summary of the model predictions made over the validation
         data by the lvl model.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

        Returns:
            dict containing saved summaries.
        '''
        return RecurrentWhisperer._load_lvl_helper(
            log_dir, run_hash, 'valid', 'summary')

    @staticmethod
    def load_hyperparameters(log_dir, run_hash):
        '''Load previously saved Hyperparameters.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

        Returns:
            dict containing the loaded hyperparameters.
        '''

        paths = RecurrentWhisperer.get_paths(log_dir, run_hash)
        hps_path = paths['hps_path']

        if os.path.exists(hps_path):
            hps_dict = Hyperparameters.restore(hps_path)
        else:
            raise IOError('%s not found.' % hps_path)

        return hps_dict

    def _step(self):
        '''Returns the number of training steps taken thus far. A step is
        typically taken with each batch of training data (although this may
        depend on subclass-specific implementation details).

        Args:
            None.

        Returns:
            step: int specifying the current training step number.
        '''
        return self.session.run(self.global_step)

    def _epoch(self):
        '''Returns the current training epoch number. An epoch is defined as
        one complete pass through the training data (i.e., multiple batches).

        Args:
            None.

        Returns:
            epoch: int specifying the current epoch number.
        '''
        return self.session.run(self.epoch)

    def _epoch_last_lvl_improvement(self):
        '''Returns the epoch of the last improvement of the lowest validation
        loss.

        Args:
            None.

        Returns:
            epoch: int specifying the epoch number.
        '''
        return self.session.run(self.epoch_last_lvl_improvement)

    def _lvl(self):
        '''Returns the lowest validation loss encountered thus far during
        training.

        Args:
            None.

        Returns:
            lvl: float specifying the lowest validation loss.
        '''
        return self.session.run(self.lvl)

    def _ltl(self):
        '''Returns the lowest training loss encountered thus far (i.e., across
        an entire pass through the training data).

        Args:
            None.

        Returns:
            ltl: float specifying the lowest training loss.
        '''
        return self.session.run(self.ltl)

    def _increment_epoch(self):
        '''Runs the TF op that increments the epoch number.

        Args:
            None.

        Returns:
            None.
        '''
        self.session.run(self.increment_epoch)

    def _np_init_weight_matrix(self, input_size, output_size):
        '''Randomly initializes a weight matrix W and bias vector b.

        For use with input data matrix X [n x input_size] and output data
        matrix Y [n x output_size], such that Y = X*W + b (with broadcast
        addition). This is the typical required usage for TF dynamic_rnn.

        Weights drawn from a standard normal distribution and are then
        rescaled to preserve input-output variance.

        Args:
            input_size: non-negative int specifying the number of input
            dimensions of the linear mapping.

            output_size: non-negative int specifying the number of output
            dimensions of the linear mapping.

        Returns:
            W: numpy array of shape [input_size x output_size] containing
            randomly initialized weights.

            b: numpy array of shape [output_size,] containing all zeros.
        '''
        if input_size == 0:
            scale = 1.0 # This avoids divide by zero error
        else:
            scale = 1.0 / np.sqrt(input_size)
        W = np.multiply(scale,self.rng.randn(input_size, output_size))
        b = np.zeros(output_size)
        return W, b

    # *************************************************************************
    # The following class methods MUST be implemented by any subclass that
    # inherits from RecurrentWhisperer.
    # *************************************************************************

    @staticmethod
    def _default_hash_hyperparameters():
        '''Defines subclass-specific default hyperparameters for the set of
        hyperparameters that are hashed to define a directory structure for
        easily managing multiple runs of the RNN training (i.e., using
        different hyperparameter settings). These hyperparameters may affect
        the model architecture or the trajectory of fitting, and as such are
        typically  swept during hyperparameter optimization.

        Values provided here will override the defaults set in this superclass
        for any hyperparameter keys that are overlapping with those in
        _DEFAULT_SUPER_HASH_HPS. Note, such overriding sets the default values
        for the subclass, and these subclass defaults can then be again
        overridden via keyword arguments input to __init__.

        Args:
            None.

        Returns:
            A dict of hyperparameters.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    @staticmethod
    def _default_non_hash_hyperparameters():
        '''Defines default hyperparameters for the set of hyperparameters that
        are NOT hashed to define a run directory. These hyperparameters should
        not influence the model architecture or the trajectory of fitting.

        Values provided here will override the defaults set in this superclass
        for any hyperparameter keys that are overlapping with those in
        _DEFAULT_SUPER_HASH_HPS. Note, such overriding sets the default values
        for the subclass, and these subclass defaults can then be again
        overridden via keyword arguments input to __init__.

        Args:
            None.

        Returns:
            A dict of hyperparameters.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

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

    def _setup_training(self, train_data, valid_data):
        '''Performs any tasks that must be completed before entering the
        training loop in self.train.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _get_data_batches(self, train_data):
        '''Randomly splits the training data into a set of batches.
        Randomization should reference the random number generator in self.rng
        so that runs can be reliably reproduced.

        Args:
            train_data: dict containing the training data.

        Returns:
            list of dicts, where each dict contains a batch of training data.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _get_batch_size(self, batch_data):
        '''Returns the number of training examples in a batch of training data.

        Args:
            batch_data: dict containing one batch of training data.

        Returns:
            int specifying the number of examples in batch_data.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _train_batch(self, batch_data):
        '''Runs one training step. This function must evaluate the Tensorboard
        summary: self.merged_opt_summary.

        Args:
            batch_data: dict containing one batch of training data. Key/value
            pairs will be specific to the subclass implementation.

        Returns:
            batch_summary: dict containing summary data from this training
            step. Minimally, this includes the following key/val pairs:

                'loss': scalar float evalutaion of the loss function over the
                data batch (i.e., an evaluation of self.loss).

                'grad_global_norm': scalar float evaluation of the norm of the
                gradient of the loss function with respect to all trainable
                variables, taken over the data batch (i.e., an evaluation of
                self.grad_global_norm).
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def predict(self, data):
        '''Runs the RNN given its inputs.

        Args:
            data: dict containing requisite data for generating predictions.
            Key/value pairs will be specific to the subclass implementation.

        Returns:
            predictions: dict containing model predictions based on data. Key/
            value pairs will be specific to the subclass implementation. Must
            contain key: 'loss' whose value is a scalar indicating the
            evaluation of the overall objective function being minimized
            during training.

            summary: dict containing high-level summaries of the predictions.
            Key/value pairs will be specific to the subclass implementation.

            If validation data are provided to train(), predictions and
            summary will be maintained in separate saved files that are
            updated each time a new lowest validation loss is achieved. By
            placing lightweight objects as values in summary (e.g., scalars),
            the summary file can be loaded faster for post-training analyses
            that do not require loading the potentially bulky predictions.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _update_valid_tensorboard(self, valid_summary):
        '''Updates the Tenorboard summaries corresponding to the validation
        data.

        Args:
            valid_summary: dict returned by predict().

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _setup_visualization(self):
        '''Sets up visualizations.

        Args:
            None.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _update_visualization(self, train_data, valid_data):
        '''Updates visualizations.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)
