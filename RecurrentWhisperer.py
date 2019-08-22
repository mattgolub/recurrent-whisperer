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
import logging
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
from Timer import Timer


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
    functions (see definitions at the end of this file for call signatures):

    _default_hash_hyperparameters()
    _default_non_hash_hyperparameters()
    _setup_model(...)
    _setup_training(...)
    _setup_visualization(...)
    _get_data_batches(...)
    _get_batch_size(...)
    _train_batch(...)
    predict(...)
    _update_visualization(...)
    _save_training_visualizations(...)
    _save_lvl_visualizations(...)
    '''

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
            _default_non_hash_hyperparameters):

                min_learning_rate: float specifying optimization termination
                criteria on the learning rate. Optimization terminates if the
                learning rate falls below this value. Default: 1e-10.

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

                min_loss: float specifying optimization termination criteria
                on the loss function evaluated across the training data (the
                epoch training loss). If None, this termination criteria is
                not applied. Default: None.

                max_train_time: float specifying the maximum amount of time
                allowed for training, expressed in seconds. If None, this
                termination criteria is not applied. Default: None.

                do_log_output: bool indicating whether to direct to a log file
                all stdout and stderr output (i.e., everything that would otherwise print to the terminal). Default: False.

                do_restart_run: bool indicating whether to force a restart of
                a training run (e.g., if a previous run with the same
                hyperparameters has saved checkpoints--the previous run will
                be deleted and restarted rather than resumed). Default: False.

                do_save_tensorboard_events: bool indicating whether or not to
                save summaries for Tensorboard monitoring. Default: True.

                do_save_ckpt: bool indicating whether or not to save model
                checkpoints. Needed because setting max_ckpt_to_keep=0 results
                in TF never deleting checkpoints. Default: True.

                do_save_lvl_ckpt: bool indicating whether or not to save model
                checkpoints specifically when a new lowest validation loss is
                achieved. Default: True.

                do_generate_training_visualizations: bool indicating whether or
                not to generate visualizations periodically throughout training.
                Frequency is controlled by n_epoochs_per_visualization_update.
                Default: True.

                do_save_training_visualizations: bool indicating whether or not
                to save the training visualizations to Tensorboard. Default:
                True.

                do_generate_lvl_visualizations: bool indicating whether or not
                to, after training is complete, load the LVL model and generate
                visualization from it. Default: True.

                do_save_lvl_visualizations: bool indicating whether or not to
                save the LVL visualizations to tensroboard. Default: True.

                do_save_lvl_train_predictions: bool indicating whether to
                maintain a .pkl file containing predictions over the training
                data based on the lowest-validation-loss parameters.

                do_save_lvl_valid_predictions: bool indicating whether to
                maintain a .pkl file containing predictions over the validation
                data based on the lowest-validation-loss parameters.

                do_save_lvl_mat_files: bool indicating whether to save .mat
                files containing predictions over the training and validation
                data each time a new lowest validation loss is achieved.
                Regardless of this setting, .pkl files are saved. Default:
                False.

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

                decive: String specifying the hardware on which to place this
                model. E.g., "gpu:0" or "gpu:0". Default: "gpu:0".

                per_process_gpu_memory_fraction: float specifying the maximum
                fraction of GPU memory to allocate. Set to None to allow
                Tensorflow to manage GPU memory. See Tensorflow documentation
                for interactions between device_count (accessed here via
                disable_gpus), enable_gpu_growth, and
                per_process_gpu_memory_fraction. Default: None.

                allow_gpu_growth: bool indicating whether to dynamically
                allocate GPU memory (True) or to monopolize the entire memory
                capacity of a GPU (False). Default: True.

                disable_gpus: bool indicating whether to disable access to any
                GPUs. Default: False.

                log_dir: string specifying the top-level directory for saving
                various training runs (where each training run is specified by
                a different set of hyperparameter settings). When tuning
                hyperparameters, log_dir is meant to be constant across
                models. Default: '/tmp/rnn_logs/'.

        Returns:
            None.
        '''
        default_hash_hps = self._integrate_hps(
            self._default_super_hash_hyperparameters(),
            self._default_hash_hyperparameters())

        default_non_hash_hps = self._integrate_hps(
            self._default_super_non_hash_hyperparameters(),
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

        self._setup_run_dir()

        with tf.device(hps.device):
            self._setup_model()
            self._setup_optimizer()
            self._maybe_setup_visualizations()

            # Each of these will create run_dir if it doesn't exist
            # (do not move above the os.path.isdir check that is in
            # _setup_run_dir)
            self._maybe_setup_tensorboard()
            self._setup_savers()

            self._setup_session()
            self._initialize_or_restore()
            self.print_trainable_variables()

    @staticmethod
    def _default_super_hash_hyperparameters():
        return {
            'random_seed': 0,
            'dtype': 'float32', # keep as string (rather than tf.float32)
                                # for better argparse handling, yaml writing
            'adam_hps': {'epsilon': 0.01},
                'alr_hps': {},
                'agnc_hps': {}}

    @staticmethod
    def _default_super_non_hash_hyperparameters():
        return {
            'min_learning_rate': 1e-10,
            'max_n_epochs': 1000,
            'max_n_epochs_without_lvl_improvement': 200,
            'min_loss': None,
            'max_train_time': None,

            'do_log_output': False,
            'do_restart_run': False,

            'do_save_tensorboard_events': True,

            'do_save_ckpt': True,
            'do_save_lvl_ckpt': True,

            'do_generate_training_visualizations': True,
            'do_save_training_visualizations': True,
            'do_generate_lvl_visualizations': True,
            'do_save_lvl_visualizations': True,

            'do_save_lvl_train_predictions': True,
            'do_save_lvl_valid_predictions': True,
            'do_save_lvl_mat_files': False,

            'max_ckpt_to_keep': 1,
            'max_lvl_ckpt_to_keep': 1,

            'n_epochs_per_ckpt': 100,
            'n_epochs_per_validation_update': 100,
            'n_epochs_per_visualization_update': 100,

            'device': 'gpu:0',
            'per_process_gpu_memory_fraction': 1.0,
            'allow_gpu_growth': True,
            'disable_gpus': False,

            'log_dir': '/tmp/rnn_logs/',
            'dataset_name': None,
            'model_name': 'RecurrentWhisperer',
            'n_folds': None,
            'fold_idx': None,}

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
    def get_hash_dir(log_dir, run_hash):
        '''Returns a path to the run_hash in the log_dir.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

        Returns:
            Path to the hash directory.
        '''
        return os.path.join(log_dir, run_hash)

    @staticmethod
    def get_run_dir(log_dir, run_hash,
        dataset_name=None, n_folds=None, fold_idx=None):
        ''' Returns a path to the direcory containing all files related to a
        given run.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.get_hash().

            dataset_name: (optional) Filesystem-safe string describing the
            dataset. If provided with all other optional arguments, this is
            systematically incorporated into the generated path.

            n_folds: (optional) Non-negative integer specifying the number of
            cross-validation folds in the run.

            fold_idx: (optional) Index specifying the cross-validation fold for
            this run.

        Returns:
            Path to the run directory.
        '''

        hash_dir = RecurrentWhisperer.get_hash_dir(log_dir, run_hash)

        if (dataset_name is not None) and \
            (n_folds is not None) and \
            (fold_idx is not None):

            fold_str = str('fold-%d-of-%d' % (fold_idx+1, n_folds))
            run_dir = os.path.join(hash_dir, dataset_name, fold_str)

            return run_dir

        else:
            return hash_dir

    @staticmethod
    def get_run_info(run_dir):
        '''Advanced functionality for models invoking K-fold cross-validation.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict with each key being a dataset name and each value is the corresponding set of cross-validation runs performed on that dataset. Jointly, these key, val pairs can reconstruct all of the "leaves" of the cross validation runs by using get_run_dir(...).

        '''

        def list_dirs(path_str):
            return [name for name in os.listdir(path_str) \
                if os.path.isdir(os.path.join(path_str, name)) ]

        run_info = {}
        if RecurrentWhisperer.is_run_dir(run_dir):
            pass
        else:
            dataset_names = list_dirs(run_dir)

            for dataset_name in dataset_names:
                cv_dir = os.path.join(run_dir, dataset_name)
                fold_names = list_dirs(cv_dir)

                run_info[dataset_name] = []
                for fold_name in fold_names:
                    run_dir = os.path.join(cv_dir, fold_name)
                    if RecurrentWhisperer.is_run_dir(run_dir):
                        run_info[dataset_name].append(fold_name)

        return run_info

    @staticmethod
    def get_paths(run_dir):
        '''Generates all paths relevant for saving and loading model data.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing all paths relevant for saving and loading model
            data. Keys are strings, with suffixes '_dir' and '_path' referring
            to directories and filenames, respectively.
        '''

        hps_dir = os.path.join(run_dir, 'hps')
        ckpt_dir = os.path.join(run_dir, 'ckpt')
        lvl_dir = os.path.join(run_dir, 'lvl')
        events_dir = os.path.join(run_dir, 'events')

        return {
            'run_dir': run_dir,

            'hps_dir': hps_dir,
            'hps_path': os.path.join(hps_dir, 'hyperparameters.pkl'),
            'hps_yaml_path': os.path.join(hps_dir, 'hyperparameters.yml'),

            'events_dir': events_dir,
            'model_log_path': os.path.join(events_dir, 'model.log'),
            'loggers_log_path': os.path.join(events_dir, 'dependencies.log'),
            'done_path': os.path.join(events_dir, 'training.done'),

            'ckpt_dir': ckpt_dir,
            'ckpt_path': os.path.join(ckpt_dir, 'checkpoint.ckpt'),

            'lvl_dir': lvl_dir,
            'lvl_ckpt_path': os.path.join(lvl_dir, 'lvl.ckpt'),

            'alr_path': os.path.join(hps_dir, 'learn_rate.pkl'),
            'agnc_path': os.path.join(hps_dir, 'norm_clip.pkl'),
            }

    @staticmethod
    def is_run_dir(run_dir):
        '''Determines whether a run exists in the filesystem.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__().

        Returns:
            bool indicating whether a run exists.
        '''

        if run_dir is None:
            return False

        paths = RecurrentWhisperer.get_paths(run_dir)

        # Check for existence of all directories that would have been created
        # if a run was executed. This won't look for various files, which may
        # or may not be saved depending on hyperparameter choices.
        for key in paths:
            if 'dir' in key and not os.path.exists(paths[key]):
                return False

        return True

    @staticmethod
    def is_done(run_dir):
        '''Determines whether a run exists in the filesystem and has run to
        completion.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__().

        Returns:
            bool indicating whether the run is "done".
        '''

        paths = RecurrentWhisperer.get_paths(run_dir)
        return os.path.exists(paths['done_path'])

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
        dataset_name = hps.dataset_name
        n_folds = hps.n_folds
        fold_idx = hps.fold_idx
        run_hash = hps.get_hash()
        run_dir = self.get_run_dir(log_dir, run_hash,
            dataset_name, n_folds, fold_idx)
        paths = self.get_paths(run_dir)

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
        self.done_path = paths['done_path']
        self.model_log_path = paths['model_log_path']
        self.loggers_log_path = paths['loggers_log_path']

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

                # Avoids pathological behavior whereby it is impossible to
                # restore a run that was started with do_restart_run = True.
                self.hps.do_restart_run = False

        if not os.path.isdir(self.run_dir):
            print('\nCreating run directory: %s.' % self.run_dir)
            os.makedirs(self.run_dir)
            os.makedirs(self.hps_dir)
            os.makedirs(self.ckpt_dir)
            os.makedirs(self.lvl_dir)
            os.makedirs(self.events_dir)

        if hps.do_log_output:
            self._setup_logger()

    def _setup_logger(self):
        '''Setup logging. Redirects (nearly) all printed output to the log file.

        Some output slips through the cracks, notably the output produced with
        calling tf.session

        Args:
            None.

        Returns:
            None.
        '''

        # Update all loggers that have been setup by dependencies
        # (e.g., Tensorflow)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_level = logging.DEBUG

        fh = logging.FileHandler(self.loggers_log_path)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)

        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.setLevel(log_level)
            for hdlr in logger.handlers:
                logger.removeHandler(hdlr)
            logger.addHandler(fh)

        # Redirect all printing, errors, and warnings generated by
        # RecurrentWhisperer
        self._default_stdout = sys.stdout
        self._default_stderr = sys.stderr
        self._log_file = open(self.model_log_path, 'a+')
        sys.stdout = self._log_file
        sys.stderr = self._log_file

    def _setup_optimizer(self):
        '''Sets up an AdamOptimizer with gradient norm clipping.

        Args:
            None.

        Returns:
            None.
        '''

        vars_to_train = tf.trainable_variables()

        with tf.name_scope('optimizer'):
            '''Maintain state using TF framework for seamless saving and
            restoring of runs'''

            self.train_timer = Timer()

            self.train_time = tf.Variable(
                0, name='train_time', trainable=False, dtype=tf.float32)
            self.train_time_placeholder = tf.placeholder(
                self.dtype, name='train_time')
            self.train_time_update = tf.assign(
                self.train_time, self.train_time_placeholder)

            self.global_step = tf.Variable(
                0, name='global_step', trainable=False, dtype=tf.int32)

            self.epoch = tf.Variable(
                0, name='epoch', trainable=False, dtype=tf.int32)

            self.increment_epoch = tf.assign_add(
                self.epoch, 1, name='increment_epoch')

            # lowest validation loss
            self.lvl = tf.Variable(
                np.inf, name='lvl', trainable=False, dtype=self.dtype)
            self.lvl_placeholder = tf.placeholder(
                self.dtype, name='lowest_validation_loss')
            self.lvl_update = tf.assign(self.lvl, self.lvl_placeholder)

            # epoch of most recent improvement in lowest validation loss
            self.epoch_last_lvl_improvement = tf.Variable(
                0, name='epoch_last_lvl_improvement',
                trainable=False, dtype=self.dtype)
            self.epoch_last_lvl_improvement_placeholder = tf.placeholder(
                self.dtype, name='epoch_last_lvl_improvement')
            self.epoch_last_lvl_improvement_update = \
                tf.assign(
                    self.epoch_last_lvl_improvement,
                    self.epoch_last_lvl_improvement_placeholder)

            # lowest training loss
            self.ltl = tf.Variable(
                np.inf, name='ltl', trainable=False, dtype=self.dtype)
            self.ltl_placeholder = tf.placeholder(
                self.dtype, name='lowest_training_loss')
            self.ltl_update = tf.assign(self.ltl, self.ltl_placeholder)

            # Gradient clipping
            grads = tf.gradients(self.loss, vars_to_train)

            self.grad_norm_clip_val = tf.placeholder(
                self.dtype, name='grad_norm_clip_val')

            clipped_grads, self.grad_global_norm = tf.clip_by_global_norm(
                grads, self.grad_norm_clip_val)

            self.clipped_grad_global_norm = tf.global_norm(clipped_grads)

            self.clipped_grad_norm_diff = \
                self.grad_global_norm - self.clipped_grad_global_norm

            zipped_grads = zip(clipped_grads, vars_to_train)

            self.learning_rate = tf.placeholder(
                self.dtype, name='learning_rate')

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, **self.hps.adam_hps)

            self.train_op = self.optimizer.apply_gradients(
                zipped_grads, global_step=self.global_step)

    def update_variables_optimized(self, vars_to_train,
        do_reset_termination_criteria=True,
        do_reset_loss_history=True,
        do_reset_learning_rate=True,
        do_reset_gradient_clipping=True):
        '''
        Updates the list of variables optimized during training. Note: this
        does not update tf.trainable_variables(), but simply updates the set
        of gradients that are computed and applied.

        Args:
            vars_to_train (optional): list of TF variables to be optimized.
            Each variable must be in tf.trainable_variables().

            do_reset_termination_criteria (optional): bool indicating whether
            to reset training termination criteria. Default: True.

            do_reset_loss_history (optional): bool indicating whether to reset
            records of the lowest training and validation losses (so that
            rescaling terms in the loss function does not upset saving model
            checkpoints.

            do_reset_learning_rate (optional): bool indicating whether to
            reset the adaptive learning rate. Default: True.

            do_reset_gradient_clipping (optional): bool indicating whether
            to reset the adaptive gradient clipping. Default: True.

        Returns:
            None.
        '''

        hps = self.hps

        if do_reset_termination_criteria:
            self._update_epoch_last_lvl_improvement(self._epoch())

        if do_reset_loss_history:
            self._update_ltl(np.inf)

            self._update_lvl(np.inf)

        if do_reset_learning_rate:
            self.adaptive_learning_rate = AdaptiveLearningRate(**hps.alr_hps)

        if do_reset_gradient_clipping:
            self.adaptive_grad_norm_clip = AdaptiveGradNormClip(**hps.agnc_hps)

        # Gradient clipping
        grads = tf.gradients(self.loss, vars_to_train)

        clipped_grads, self.grad_global_norm = tf.clip_by_global_norm(
            grads, self.grad_norm_clip_val)

        self.clipped_grad_global_norm = tf.global_norm(clipped_grads)

        self.clipped_grad_norm_diff = \
            self.grad_global_norm - self.clipped_grad_global_norm

        zipped_grads = zip(clipped_grads, vars_to_train)

        self.train_op = self.optimizer.apply_gradients(
            zipped_grads, global_step=self.global_step)

    def _maybe_setup_visualizations(self):

        if self.hps.do_generate_training_visualizations or \
            self.hps.do_generate_lvl_visualizations:

            self._setup_visualizations()

    def _maybe_setup_tensorboard(self):
        '''Sets up the Tensorboard FileWriter and Tensorboard summaries for
        monitoring the optimization.

        Args:
            None.

        Returns:
            None.
        '''

        if self.hps.do_save_tensorboard_events:

            self.writer = tf.summary.FileWriter(self.events_dir)
            self.writer.add_graph(tf.get_default_graph())

            # Tensorboard summaries (updated during training via _train_epoch)
            with tf.name_scope('tb-optimizer'):
                summaries = []
                summaries.append(tf.summary.scalar('loss', self.loss))
                summaries.append(tf.summary.scalar('lvl', self.lvl))
                summaries.append(tf.summary.scalar(
                    'learning rate',
                    self.learning_rate))
                summaries.append(tf.summary.scalar(
                    'grad global norm',
                    self.grad_global_norm))
                summaries.append(tf.summary.scalar(
                    'grad norm clip val',
                    self.grad_norm_clip_val))
                summaries.append(tf.summary.scalar(
                    'clipped grad global norm',
                    self.clipped_grad_global_norm))
                summaries.append(tf.summary.scalar(
                    'grad clip diff',
                    self.clipped_grad_norm_diff))

                self.merged_opt_summary = tf.summary.merge(summaries)

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
        config.allow_soft_placement = True
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

            # Start training timer from scratch
            self.train_time_offset = 0.0
        else:
            self._restore_from_checkpoint(ckpt)

            # Resume training timer from value at last save.
            self.train_time_offset = self.session.run(self.train_time)

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

        N_EPOCH_SPLITS = 6

        self._setup_training(train_data, valid_data)

        if self._is_training_complete(self._ltl()):
            # If restoring from a completed run, do not enter training loop
            # and do not save a new checkpoint.
            return

        self.train_timer.start()

        # Training loop
        print('Entering training loop.')
        while True:
            epoch_timer = Timer(N_EPOCH_SPLITS, n_indent=1, name='Epoch')
            epoch_timer.start()

            # *****************************************************************

            data_batches = self._get_data_batches(train_data)

            epoch_timer.split('data')

            # *****************************************************************

            epoch_loss = self._train_epoch(data_batches)

            epoch_timer.split('train')

            # *****************************************************************

            self._maybe_update_validation(train_data, valid_data)

            epoch_timer.split('validation')

            # *****************************************************************

            self._maybe_update_training_visualizations(train_data, valid_data)

            epoch_timer.split('visualize')

            # *****************************************************************

            self._maybe_save_checkpoint()

            epoch_timer.split('save')

            # *****************************************************************

            if self._is_training_complete(epoch_loss, valid_data is not None):
                break

            epoch_timer.split('other')
            epoch_timer.disp()

        self._close_training(train_data, valid_data)

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
            self._update_ltl(epoch_loss)

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

        if self.hps.do_save_tensorboard_events:
            self._update_valid_tensorboard(summary)

        self._maybe_save_lvl_checkpoint(
            summary['loss'], train_data, valid_data)

    def _maybe_update_training_visualizations(self, train_data, valid_data):
        '''Updates visualizations if the current epoch number indicates that
        an update is due.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''

        hps = self.hps
        if hps.do_generate_training_visualizations and \
            np.mod(self._epoch(), hps.n_epochs_per_visualization_update) == 0:

            self._update_visualizations(train_data, valid_data)

            if hps.do_save_training_visualizations:

                self._save_training_visualizations()

    def _maybe_save_checkpoint(self):
        '''Saves a model checkpoint if the current epoch number indicates that
        a checkpoint is due.

        Args:
            None.

        Returns:
            None.
        '''
        if self.hps.do_save_ckpt and \
            np.mod(self._epoch(), self.hps.n_epochs_per_ckpt) == 0:

            self._update_train_time()
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

        Note that the hyperparameters files are not updated from their original
        form. This can become relevant if restoring a model and resuming
        training using updated values of non-hash hyperparameters.

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
        lower than all previously evaluated validation loss values. This
        includes saving model predictions over the traiing and validation data.

        If prediction summaries are generated, those summaries are saved in
        separate .pkl files (and optional .mat files). See docstring for
        predict() for additional detail.

        Args:
            valid_loss: float indicating the current loss evaluated over the
            validation data.

            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        if (self._epoch()==0 or valid_loss < self._lvl()):

            print('Achieved lowest validation loss.')

            self._update_lvl(valid_loss)
            self._update_epoch_last_lvl_improvement(self._epoch())

            if self.hps.do_save_lvl_ckpt:
                self._update_train_time()
                print('\tSaving checkpoint.')
                self.lvl_saver.save(self.session,
                    self.lvl_ckpt_path, global_step=self._step())

            train_pred, train_summary = self.predict(train_data)
            self._maybe_save_lvl_predictions(train_pred, 'train')
            self._maybe_save_lvl_summaries(train_summary, 'train')

            valid_pred, valid_summary = self.predict(valid_data)
            self._maybe_save_lvl_predictions(valid_pred, 'valid')
            self._maybe_save_lvl_summaries(valid_summary, 'valid')

    def restore_from_lvl_checkpoint(self, model_checkpoint_path=None):
        '''Restores a model from a previously saved lowest-validation-loss
        checkpoint.

        Args:
            model_checkpoint_path (optional): string containing a path to
            a model checkpoint. Use this as an override if needed for
            loading models that were saved under a different directory
            structure (e.g., on another machine).

        Returns:
            None.

        Raises:
            FileNotFoundError (if no lowest-validation-loss checkpoint exists).
        '''
        if model_checkpoint_path is None:
            lvl_ckpt = tf.train.get_checkpoint_state(self.lvl_dir)
            model_checkpoint_path = lvl_ckpt.model_checkpoint_path

        if not(tf.train.checkpoint_exists(model_checkpoint_path)):
            raise FileNotFoundError('Checkpoint does not exist: %s'
                                    % model_checkpoint_path)

        # Restore previous session
        print('\tLoading lvl checkpoint: %s.'
              % ntpath.basename(model_checkpoint_path))
        self.lvl_saver.restore(self.session, model_checkpoint_path)

    def _maybe_save_lvl_predictions(self, predictions, train_or_valid_str):
        '''Saves all model predictions in .pkl files (and optionally in .mat
        files as well).

        Args:
            predictions: dict containing model predictions.

            train_or_valid_str: either 'train' or 'valid', indicating whether
            data contains training data or validation data, respectively.

        Returns:
            None.
        '''

        if predictions is not None:
            if (self.hps.do_save_lvl_train_predictions and
                train_or_valid_str == 'train') or \
                (self.hps.do_save_lvl_valid_predictions and
                train_or_valid_str == 'valid'):

                print('\tSaving lvl predictions (%s).' % train_or_valid_str)
                filename_no_extension = train_or_valid_str + '_predictions'
                self._save_helper(predictions, filename_no_extension)

    def _maybe_save_lvl_summaries(self,
        summary, train_or_valid_str):

        if summary is not None:
            print('\tSaving lvl summary (%s).' % train_or_valid_str)
            # E.g., train_predictions or valid_summary
            filename_no_extension = train_or_valid_str + '_summary'
            self._save_helper(summary, filename_no_extension)

    def _save_done_file(self):
        '''Save an empty .done file (indicating that the training procedure
        ran to completion.

        Args:
            None.

        Returns:
            None.
        '''
        print('\tSaving .done file...')

        save_path = self.done_path
        file = open(save_path, 'wb')
        file.write('')
        file.close()

    def _save_helper(self, data_to_save, filename_no_extension):
        '''Pickle and save data as .pkl file. Optionally also save the data as
        a .mat file.

            Args:
                data_to_save: any pickle-able object to be pickled and saved.

            Returns:
                None.
        '''

        def save_pkl(data_to_save, save_path_no_extension):
            '''Pickle and save data as .pkl file.

            Args:
                data_to_save: any pickle-able object to be pickled and saved.

                save_path_no_extension: path at which to save the data,
                including an extensionless filename.

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

        pkl_path = os.path.join(self.lvl_dir, filename_no_extension)
        save_pkl(data_to_save, pkl_path)

        if self.hps.do_save_lvl_mat_files:
            mat_path = os.path.join(self.lvl_dir, filename_no_extension)
            save_mat(data_to_save, pkl_path)

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

        if loss is np.inf:
            print('\nStopping optimization: loss is Inf!')
            return True

        if np.isnan(loss):
            print('\nStopping optimization: loss is NaN!')
            return True

        if hps.min_loss is not None and loss <= hps.min_loss:
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

        if hps.max_train_time is not None and \
            self._get_train_time() > hps.max_train_time:

            print ('\nStopping optimization: training time exceeds '
                'maximum allowed.')
            return True

        if do_check_lvl and \
            self._epoch() - self._epoch_last_lvl_improvement() >= \
                hps.max_n_epochs_without_lvl_improvement:

            print('\nStopping optimization:'
                  ' reached maximum number of training epochs'
                  ' without improvement to the lowest validation loss.')
            return True

        return False

    def _close_training(self, train_data=None, valid_data=None):
        ''' Optionally saves a final checkpoint, then loads the LVL model and
        generates LVL visualizations.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.

        Raises:
            Warning if attempting to generate LVL visualizations when no LVL
            checkpoint was saved.
        '''

        # Save checkpoint upon completing training
        if self.hps.do_save_ckpt:
            self._save_checkpoint()

        if self.hps.do_generate_lvl_visualizations:
            if self.hps.do_save_lvl_ckpt:
                # Generate LVL visualizations
                print('\tGenerating visualizations from restored LVL model...')
                self.restore_from_lvl_checkpoint()
                self._update_visualizations(train_data, valid_data)

                if self.hps.do_save_lvl_visualizations:
                    self._save_lvl_visualizations()
            else:
                raise Warning('Attempted to generate LVL visualizations, '
                    'but cannot because no LVL model checkpoint was saved.')

        self._save_done_file()

        if self.hps.do_log_output:
            self._log_file.close()

            # Redirect all printing, errors, and warnings back to defaults
            sys.stdout = self._default_stdout
            sys.stderr = self._default_stderr

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

    def _get_train_time(self):
        '''Returns the time elapsed during training, measured in seconds, and
        accounting for restoring from previously saved runs.

        Args:
            None.

        Returns:
            float indicating time elapsed during training..
        '''
        return self.train_time_offset + self.train_timer()

    def _update_train_time(self):
        '''Runs the TF op that updates the time elapsed during training.

        Args:
            None.

        Returns:
            None.
        '''
        time_val = self._get_train_time()
        self.session.run(
            self.train_time_update,
            feed_dict={self.train_time_placeholder: time_val})

    def _increment_epoch(self):
        '''Runs the TF op that increments the epoch number.

        Args:
            None.

        Returns:
            None.
        '''
        self.session.run(self.increment_epoch)

    def _update_ltl(self, ltl):
        '''Runs the TF op that updates the lowest training loss.

        Args:
            ltl: A numpy scalar value indicating the (new) lowest training loss.

        Returns:
            None.
        '''

        self.session.run(
            self.ltl_update,
            feed_dict={self.ltl_placeholder: ltl})

    def _update_lvl(self, lvl):
        '''Runs the TF op that updates the lowest validation loss.

        Args:
            lvl: A numpy scalar value indicating the (new) lowest validation
            loss.

        Returns:
            None.
        '''
        self.session.run(
            self.lvl_update,
            feed_dict={self.lvl_placeholder: lvl})

    def _update_epoch_last_lvl_improvement(self, epoch):
        '''Runs the TF op that updates the counter that tracks the most recent
        improvement in the lowest validation loss.

        Args:
            epoch: A numpy scalar value indicating the epoch of the most recent improvement in the lowest validation loss.

        Returns:
            None.
        '''
        self.session.run(
            self.epoch_last_lvl_improvement_update,
            feed_dict={self.epoch_last_lvl_improvement_placeholder: epoch})

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

    def get_n_params(self, scope=None):
        ''' Counts the number of trainable parameters in a Tensorflow model
        (or scope within a model).

        Args:
            scope (optional): string specifying optional scope in which to
            count parameters. See docstring for tf.trainable_variables.

        Returns:
            integer specifying the number of trainable parameters.
        '''

        n_params = sum([np.prod(v.shape).value \
            for v in tf.trainable_variables(scope)])

        return n_params

    @staticmethod
    def _get_lvl_path(run_dir, train_or_valid_str, predictions_or_summary_str):
        ''' Builds paths to the various files saved when the model achieves a
        new lowest validation loss (lvl).

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            train_or_valid_str: either 'train' or 'valid', indicating whether
            to load predictions/summary from the training data or validation
            data, respectively.

            predictions_or_summary_str: either 'predictions' or 'summary',
            indicating whether to load model predictions or summaries thereof,
            respectively.

        Returns:
            string containing the path to the desired file.
        '''

        paths = RecurrentWhisperer.get_paths(run_dir)
        filename = train_or_valid_str + '_' + \
            predictions_or_summary_str + '.pkl'
        path_to_file = os.path.join(paths['lvl_dir'], filename)

        return path_to_file

    @staticmethod
    def _load_lvl_helper(
        run_dir,
        train_or_valid_str,
        predictions_or_summary_str):
        '''Loads previously saved model predictions or summaries thereof.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            train_or_valid_str: either 'train' or 'valid', indicating whether
            to load predictions/summary from the training data or validation
            data, respectively.

            predictions_or_summary_str: either 'predictions' or 'summary',
            indicating whether to load model predictions or summaries thereof,
            respectively.

        Returns:
            dict containing saved data.
        '''

        path_to_file = RecurrentWhisperer._get_lvl_path(
            run_dir, train_or_valid_str, predictions_or_summary_str)

        if os.path.exists(path_to_file):
            file = open(path_to_file, 'rb')
            load_path = file.read()
            data = cPickle.loads(load_path)
            file.close()
        else:
            raise IOError('%s not found.' % path_to_file)

        return data

    @staticmethod
    def load_lvl_train_predictions(run_dir):
        '''Loads all model predictions made over the training data by the lvl
        model.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing saved predictions.
        '''
        return RecurrentWhisperer._load_lvl_helper(
            run_dir, 'train', 'predictions')

    @staticmethod
    def load_lvl_train_summary(run_dir):
        '''Loads summary of the model predictions made over the training data
        by the lvl model.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing saved summaries.
        '''
        return RecurrentWhisperer._load_lvl_helper(
            run_dir, 'train', 'summary')

    @staticmethod
    def load_lvl_valid_predictions(run_dir):
        '''Loads all model predictions from train_predictions.pkl.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            train_predictions:
                dict containing saved predictions on the training data by the
                lvl model.
        '''

        return RecurrentWhisperer._load_lvl_helper(
            run_dir, 'valid', 'predictions')

    @staticmethod
    def load_lvl_valid_summary(run_dir):
        '''Loads summary of the model predictions made over the validation
         data by the lvl model.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing saved summaries.
        '''
        return RecurrentWhisperer._load_lvl_helper(
            run_dir, 'valid', 'summary')

    @staticmethod
    def load_hyperparameters(run_dir):
        '''Load previously saved Hyperparameters.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing the loaded hyperparameters.
        '''

        paths = RecurrentWhisperer.get_paths(run_dir)

        hps_path = paths['hps_path']

        if os.path.exists(hps_path):
            hps_dict = Hyperparameters.restore(hps_path)
        else:
            raise IOError('%s not found.' % hps_path)

        return hps_dict

    def print_trainable_variables(self):
        '''Prints the current set of trainable variables.

        Args:
            None.

        Returns:
            None.
        '''
    	print('\nTrainable variables:')
    	for v in tf.trainable_variables():
    		print('\t' + v.name + ': ' + str(v.shape))
    	print('')

    # *************************************************************************
    # The following class methods MUST be implemented by any subclass that
    # inherits from RecurrentWhisperer.
    # *************************************************************************

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
        _default_super_non_hash_hyperparameters. Note, such overriding sets the
        default values for the subclass, and these subclass defaults can then
        be again overridden via keyword arguments input to __init__.

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

                'loss': scalar float evaluation of the loss function over the
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
        '''Runs the model given its inputs.

        If validation data are provided to train(), predictions are optionally
        saved each time a new lowest validation loss is achieved.

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
        '''Updates the Tensorboard summaries corresponding to the validation
        data. Only called if do_save_tensorboard_events.

        Args:
            valid_summary: dict returned by predict().

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _setup_visualizations(self):
        '''Sets up visualizations. Only called if
            do_generate_training_visualizations or
            do_generate_lvl_visualizations.

        Args:
            None.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _update_visualizations(self, train_data, valid_data):
        '''Updates visualizations. Only called if
            do_generate_training_visualizations OR
            do_generate_lvl_visualizations.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _save_training_visualizations(self):
        ''' Saves training visualizations. Only called if
            do_generate_training_visualizations AND
            do_save_training_visualizations.

        Args:
            None.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _save_lvl_visualizations(self):
        ''' Saves lowest-validation-loss visualizations. Only called ALL of the
        following hyperparameters are True:
            do_save_lvl_ckpt
            do_generate_lvl_visualizations
            do_save_lvl_visualizations.

        Args:
            None.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)
