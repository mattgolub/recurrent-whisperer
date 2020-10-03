'''
RecurrentWhisperer.py
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
from subprocess import call
import pdb

import tensorflow as tf
import numpy as np
import numpy.random as npr
import cPickle
import scipy.io as spio

if os.environ.get('DISPLAY','') == '':
    # Ensures smooth running across environments, including servers without
    # graphical backends.
    print('No display found. Using non-interactive Agg backend.')
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    _train_batch(...)
    _predict_batch(...)
    _get_batch_size(...)
    _split_data_into_batches(...)
    _subselect_batch(...)
    _combine_prediction_batches(...)
    _update_valid_tensorboard_summaries
    _update_visualization(...)
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

                max_batch_size: int specifying the size of the largest batch
                to create during training / prediction. Data are batched into
                roughly equal sized batches, depending on whether the number
                of trials divides evenly by this number.

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

                name: string describing this instance of RecurrentWhisperer.
                Used for scoping and uniquifying of TF variables.
                Default: 'rw'.

                mode: string identifying the mode in which the model will be
                used. This is never used internally, and is only included for
                optional use by external run scripts. This is included here
                to simplify command-line argument parsing, which is already
                nicely handled by Hyperparameters.py Default: 'train'.

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
                all stdout and stderr output (i.e., everything that would
                otherwise print to the terminal). Default: False.

                do_restart_run: bool indicating whether to force a restart of
                a training run (e.g., if a previous run with the same
                hyperparameters has saved checkpoints--the previous run will
                be deleted and restarted rather than resumed). Default: False.

                do_save_tensorboard_summaries: bool indicating whether or not
                to save summaries to Tensorboard. Default: True.

                do_save_tensorboard_histograms: bool indicating whether or not
                to save histograms of each trained variable to Tensorboard
                throughout training. Default: True.

                do_save_tensorboard_images: bool indicating whether or not to
                save visualizations to Tensorboard Images. Default: True.

                do_save_ckpt: bool indicating whether or not to save model
                checkpoints. Needed because setting max_ckpt_to_keep=0 results
                in TF never deleting checkpoints. Default: True.

                do_save_lvl_ckpt: bool indicating whether or not to save model
                checkpoints specifically when a new lowest validation loss is
                achieved. Default: True.

                do_generate_training_visualizations: bool indicating whether or
                not to generate visualizations periodically throughout
                training. Frequency is controlled by
                n_epoochs_per_visualization_update. Default: True.

                do_save_training_visualizations: bool indicating whether or not
                to save the training visualizations to Tensorboard. Default:
                True.

                do_generate_lvl_visualizations: bool indicating whether or not
                to, after training is complete, load the LVL model and generate
                visualization from it. Default: True.

                do_save_lvl_visualizations: bool indicating whether or not to
                save the LVL visualizations to Tensorboard. Default: True.

                do_save_lvl_train_predictions: bool indicating whether to
                maintain a .pkl file containing predictions over the training
                data based on the lowest-validation-loss parameters.

                do_save_lvl_train_summaries: bool indicating whether to
                maintain a .pkl file containing summaries of the training
                predictions based on the lowest-validation-loss parameters.

                do_save_lvl_valid_predictions: bool indicating whether to
                maintain a .pkl file containing predictions over the validation
                data based on the lowest-validation-loss parameters.

                do_save_lvl_valid_summaries: bool indicating whether to
                maintain a .pkl file containing summaries of the validation
                 predictions based on the lowest-validation-loss parameters.

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

                device: String specifying the hardware on which to place this
                model. E.g., "gpu:0" or "gpu:1". Default: "gpu:0".

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

        hps = self.setup_hps(kwargs)

        self.hps = hps
        self.dtype = getattr(tf, hps.dtype)

        '''Make parameter initializations and data batching reproducible
        across runs.'''
        self.rng = npr.RandomState(hps.random_seed)
        tf.set_random_seed(hps.random_seed)
        ''' Note: Currently this state will not transfer across saves and
        restores. Thus behavior will only be reproducible for uninterrupted
        runs (i.e., that do not require restoring from a checkpoint). The fix
        would be to draw all random numbers needed for a run upon starting or
        restoring a run.'''

        self.prev_loss = None
        self.adaptive_learning_rate = AdaptiveLearningRate(**hps.alr_hps)
        self.adaptive_grad_norm_clip = AdaptiveGradNormClip(**hps.agnc_hps)

        self._setup_run_dir()

        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print('\n\nCUDA_VISIBLE_DEVICES: %s'
                % str(os.environ['CUDA_VISIBLE_DEVICES']))
        print('Building TF model on %s\n' % hps.device)

        with tf.device(hps.device):

            with tf.variable_scope(hps.name, reuse=tf.AUTO_REUSE):

                self._setup_records()
                self._setup_model()
                self._setup_optimizer()

                self._maybe_setup_visualizations()

                # Each of these will create run_dir if it doesn't exist
                # (do not move above the os.path.isdir check that is in
                # _setup_run_dir)
                self._maybe_setup_tensorboard_summaries()
                self._setup_savers()

                self._setup_session()

                if not hps.do_custom_restore:
                    self._initialize_or_restore()
                    self.print_trainable_variables()

    # *************************************************************************
    # Static access ***********************************************************
    # *************************************************************************

    @staticmethod
    def default_hyperparameters(subclass):
        ''' Returns the dict of ALL (RecurrentWhisperer + subclass)
        hyperparameters (both hash and non-hash). This is needed for
        command-line argument parsing.

        Args:
            subclass: the __class__ of the object that inherits from
            RecurrentWhisperer. This provides static access to that subclass'
            default hyperparameters.

        Returns:
            dict of hyperparameters.

        To Do: Convert this and related functions to @classmethod.
        '''

        hps = \
            RecurrentWhisperer.default_hash_hyperparameters(subclass)
        non_hash_hps = \
            RecurrentWhisperer.default_non_hash_hyperparameters(subclass)

        hps.update(non_hash_hps)

        return hps

    @staticmethod
    def default_hash_hyperparameters(subclass):
        ''' Returns the dict of ALL (RecurrentWhisperer + subclass)
        hyperparameters that are included in the run hash.

        Args:
            subclass: the __class__ of the object that inherits from
            RecurrentWhisperer. This provides static access to that subclass'
            default hyperparameters.

        Returns:
            dict of hyperparameters.
        '''

        hash_hps = Hyperparameters.integrate_hps(
            RecurrentWhisperer._default_super_hash_hyperparameters(),
            subclass._default_hash_hyperparameters())

        return hash_hps

    @staticmethod
    def default_non_hash_hyperparameters(subclass):
        ''' Returns the dict of ALL (RecurrentWhisperer + subclass)
        hyperparameters that are NOT included in the run hash.

        Args:
            subclass: the __class__ of the object that inherits from
            RecurrentWhisperer. This provides static access to that subclass'
            default hyperparameters.

        Returns:
            dict of hyperparameters.
        '''

        non_hash_hps = Hyperparameters.integrate_hps(
            RecurrentWhisperer._default_super_non_hash_hyperparameters(),
            subclass._default_non_hash_hyperparameters())
        return non_hash_hps

    @staticmethod
    def _default_super_hash_hyperparameters():
        ''' Returns the dict of RecurrentWhisperer hyperparameters that are
        included in the run hash.

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''

        ''' To allow subclasses to effectively manage HPs, this should include
        all HPs for all helper classes (i.e., '*_hps')--not just those that
        are changed from their defaults. '''
        return {
            'max_batch_size': 256,
            'random_seed': 0,
            'dtype': 'float32', # keep as string (rather than tf.float32)
                                # for better argparse handling, yaml writing
            'adam_hps': {
                'epsilon': 0.01,
                'beta1': 0.9,
                'beta2': 0.999,
                'use_locking': False,
                'name': 'Adam'
                },
            'alr_hps': AdaptiveLearningRate.default_hps,
            'agnc_hps': AdaptiveGradNormClip.default_hps,
            }

    @staticmethod
    def _default_super_non_hash_hyperparameters():
        ''' Returns the dict of RecurrentWhisperer hyperparameters that are
        NOT included in the run hash.

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''

        # See comment in _default_super_hash_hyperparameters()
        return {
            'name': 'RecurrentWhisperer',
            'mode': 'train',
            'max_n_epochs_without_lvl_improvement': 200,
            'min_loss': None,
            'max_train_time': None,

            'do_log_output': False,
            'do_restart_run': False,
            'do_custom_restore': False,

            'do_save_tensorboard_summaries': True,
			'do_save_tensorboard_histograms': True,
            'do_save_tensorboard_images': True,

            'do_save_ckpt': True,
            'do_save_lvl_ckpt': True,

            'do_generate_training_visualizations': True,
            'do_save_training_visualizations': True,

            'do_generate_final_visualizations': True,
            'do_save_final_visualizations': True,

            'do_generate_lvl_visualizations': True,
            'do_save_lvl_visualizations': True,

            'do_save_lvl_train_predictions': True,
            'do_save_lvl_train_summaries': True,
            'do_save_lvl_valid_predictions': True,
            'do_save_lvl_valid_summaries': True,
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
            'n_folds': None,
            'fold_idx': None,}

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

    @classmethod
    def setup_hps(cls, hps_dict):

        return Hyperparameters(hps_dict,
            cls.default_hash_hyperparameters(cls),
            cls.default_non_hash_hyperparameters(cls))

    @staticmethod
    def get_hash_dir(log_dir, run_hash):
        '''Returns a path to the run_hash in the log_dir.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.run_hash.

        Returns:
            Path to the hash directory.
        '''
        return os.path.join(log_dir, run_hash)

    @staticmethod
    def get_run_dir(log_dir, run_hash, n_folds=None, fold_idx=None):
        ''' Returns a path to the directory containing all files related to a
        given run.

        Args:
            log_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            run_hash: string containing the hyperparameters hash used to
            establish the run directory. Returned by
            Hyperparameters.run_hash.

            n_folds: (optional) Non-negative integer specifying the number of
            cross-validation folds in the run.

            fold_idx: (optional) Index specifying the cross-validation fold for
            this run.

        Returns:
            Path to the run directory.
        '''

        hash_dir = RecurrentWhisperer.get_hash_dir(log_dir, run_hash)

        if (n_folds is not None) and (fold_idx is not None):

            fold_str = str('fold-%d-of-%d' % (fold_idx+1, n_folds))
            run_dir = os.path.join(hash_dir, fold_str)

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
            fold_names = list_dirs(run_dir)

            run_info = []
            for fold_name in fold_names:
                fold_dir = os.path.join(run_dir, fold_name)
                if RecurrentWhisperer.is_run_dir(fold_dir):
                    run_info.append(fold_name)

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
        fig_dir = os.path.join(run_dir, 'figs')
        fp_dir = os.path.join(run_dir, 'fps')

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

            'fig_dir': fig_dir,
            'fp_dir': fp_dir,
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

    @staticmethod
    def get_command_line_call(run_script,
                              hp_dict={},
                              do_format_for_bash=False):

        ''' Generates a command line call to a user-specified bash script with
        RecurrentWhisperer hyperparameters passed in as command-line arguments.
        Can be formatted for execution within Python  or from a terminal /
        bash script.

        Args:
            run_script: string specifying the bash script call,
            e.g., 'location/of/your/run_script.sh'

            hp_dict: (optional) dict containing any hps to override defaults.
            Default: {}

            do_format_for_bash: (optional) bool indicating whether to return
            the command-line call as a string (for writing into a higher-level
            bash script; for copying into a terminal). Default: False (see
            below).

        Returns:
            Default:
                cmd_list: a list that is interpretable by subprocess.call:
                    subprocess.call(cmd_list)

            do_format_for_bash == True:
                cmd_str: a string (suitable for placing in a bash script or
                copying into a terminal .
        '''

        def raise_error():
        	# This should not be reachable--Hyperparameters.flatted converts to
        	# a colon delimited format.
        	raise ValueError('HPs that are themselves dicts are not supported')

        flat_hps = Hyperparameters.flatten(hp_dict)
        hp_names = flat_hps.keys()
        hp_names.sort()

        if do_format_for_bash:

            cmd_str = 'python ' + run_script
            for hp_name in hp_names:
                val = flat_hps[hp_name]
                if isinstance(val, dict):
                    omit_dict_hp(hp_name)
                else:
                    cmd_str += str(' --%s=%s' % (hp_name, str(val)))

            return cmd_str

        else:

            cmd_list = ['python', run_script]
            for hp_name in hp_names:
                val = flat_hps[hp_name]
                if isinstance(val, dict):
                    omit_dict_hp(hp_name)
                else:
                    cmd_list.append(str('--%s' % hp_name))
                    str_val = str(val)

                    # negative numbers misinterpreted by argparse as optional arg. This extra-space hack gets around it.
                    if str_val[0] == '-':
                        str_val = ' ' + str_val

                    cmd_list.append(str_val)

            return cmd_list

    @staticmethod
    def execute_command_line_call(run_script, hp_dict={}):
        ''' Executes a command line call to a user-specified bash script with
        RecurrentWhisperer hyperparameters passed in as command-line arguments.

        Args:
            run_script: string specifying the bash script call,
            e.g., 'location/of/your/run_script.sh'

            hp_dict: (optional) dict containing any hps to override defaults.
            Default: {}

        Returns:
            None.
        '''

        cmd_list = RecurrentWhisperer.get_command_line_call(
            run_script, hp_dict)
        print(cmd_list)
        call(cmd_list)

    # *************************************************************************
    # Setup *******************************************************************
    # *************************************************************************

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
        n_folds = hps.n_folds
        fold_idx = hps.fold_idx
        run_hash = hps.hash
        run_dir = self.get_run_dir(log_dir, run_hash, n_folds, fold_idx)
        paths = self.get_paths(run_dir)

        self.run_hash = run_hash
        self.run_dir = paths['run_dir']
        self.hps_dir = paths['hps_dir']
        self.hps_path = paths['hps_path']
        self.hps_yaml_path = paths['hps_yaml_path']
        self.fig_dir = paths['fig_dir']
        self.fp_dir = paths['fp_dir']
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
            lvl_ckpt = tf.train.get_checkpoint_state(self.lvl_dir)
            if ckpt is None and lvl_ckpt is None:
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
            os.makedirs(self.fig_dir)
            os.makedirs(self.fp_dir)

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

    def _setup_records(self):
        '''Sets up basic record keeping for training steps, epochs, timing,
        and lowest training/validation losses.

        Args:
            None.

        Returns:
            None.
        '''

        with tf.variable_scope('records', reuse=False):
            '''Maintain state using TF framework for seamless saving and
            restoring of runs'''

            self.train_timer = Timer()

            ''' Counter to track the current training epoch number. An epoch
            is defined as one complete pass through the training data (i.e.,
            multiple batches).'''
            self.epoch = tf.Variable(
                0, name='epoch', trainable=False, dtype=tf.int32)
            self.increment_epoch = tf.assign_add(
                self.epoch, 1, name='increment_epoch')

            self.train_time = tf.Variable(
                0, name='train_time', trainable=False, dtype=self.dtype)
            self.train_time_placeholder = tf.placeholder(
                self.dtype, name='train_time')
            self.train_time_update = tf.assign(
                self.train_time, self.train_time_placeholder)

            self.global_step = tf.Variable(
                0, name='global_step', trainable=False, dtype=tf.int32)

            # lowest validation loss
            self.lvl = tf.Variable(
                np.inf, name='lvl', trainable=False, dtype=self.dtype)
            self.lvl_placeholder = tf.placeholder(
                self.dtype, name='lowest_validation_loss')
            self.lvl_update = tf.assign(self.lvl, self.lvl_placeholder)
            self.epoch_last_lvl_improvement = 0

            # lowest training loss
            self.ltl = tf.Variable(
                np.inf, name='ltl', trainable=False, dtype=self.dtype)
            self.ltl_placeholder = tf.placeholder(
                self.dtype, name='lowest_training_loss')
            self.ltl_update = tf.assign(self.ltl, self.ltl_placeholder)

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

    def _setup_optimizer(self):
        '''Sets up an AdamOptimizer with gradient norm clipping.

        Args:
            None.

        Returns:
            None.
        '''

        vars_to_train = self.trainable_variables

        with tf.variable_scope('optimizer', reuse=False):

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

    def _maybe_setup_visualizations(self):

        if self.hps.do_generate_training_visualizations or \
            self.hps.do_generate_lvl_visualizations or \
            self.hps.do_generate_final_visualizations:

            self._setup_visualizations()

    def _setup_visualizations(self):
        '''Sets up visualizations. Only called if
            do_generate_training_visualizations or
            do_generate_lvl_visualizations.

        Args:
            None.

        Returns:
            figs: dict with string figure names as keys and matplotlib.pyplot.figure objects as values. Typical usage will
            populate this dict upon the first call to update_visualizations().
        '''
        self.figs = dict()

    def _setup_savers(self):
        '''Sets up Tensorflow checkpoint saving.

        Args:
            None.

        Returns:
            None.
        '''

        self.savers = dict()

        # save every so often
        self.savers['seso'] = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.hps.max_ckpt_to_keep)

        # lowest validation loss
        self.savers['lvl'] = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.hps.max_lvl_ckpt_to_keep)

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

    # *************************************************************************
    # Initializations *********************************************************
    # *************************************************************************

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
        lvl_ckpt = tf.train.get_checkpoint_state(self.lvl_dir)

        if ckpt is not None:
            self._restore_from_checkpoint(self.savers['seso'], self.ckpt_dir)
        elif lvl_ckpt is not None:
            self._restore_from_checkpoint(self.savers['lvl'], self.lvl_dir)
        else:
            # Initialize new session
            print('Initializing new run (%s).' % self.hps.hash)
            self.session.run(tf.global_variables_initializer())

            self.hps.save_yaml(self.hps_yaml_path) # For visual inspection
            self.hps.save(self.hps_path) # For restoring a run via its run_dir
            # (i.e., without needing to manually specify hps)

            # Start training timer from scratch
            self.train_time_offset = 0.0

    # *************************************************************************
    # Tensorboard *************************************************************
    # *************************************************************************

    def _maybe_setup_tensorboard_summaries(self):
        '''Sets up the Tensorboard FileWriter and graph. Optionally sets up
        Tensorboard Summaries. Tensorboard Images are not setup here (see
        _update_tensorboard_images()).

        Args:
            None.

        Returns:
            None.
        '''

        self.tensorboard = {}

        self.tensorboard['writer'] = tf.summary.FileWriter(self.events_dir)
        self.tensorboard['writer'].add_graph(tf.get_default_graph())

        if self.hps.do_save_tensorboard_summaries:
            self._setup_tensorboard_summaries()

    def _setup_tensorboard_summaries(self):
        '''Sets up Tensorboard summaries for monitoring the optimization.

        Args:
            None.

        Returns:
            None.
        '''

        if self.hps.do_save_tensorboard_histograms:
            hist_ops = {}
            for v in self.trainable_variables:
                hist_ops[v.name] = v
            self.tensorboard['merged_hist_summary'] = \
                self._build_merged_tensorboard_summaries(
                    scope='model',
                    ops_dict=hist_ops,
                    summary_fcn=tf.summary.histogram)

        self.tensorboard['merged_opt_summary'] = \
            self._build_merged_tensorboard_summaries(
                scope='tb-optimizer',
                ops_dict={
                    'loss': self.loss,
                    'lvl': self.lvl,
                    'learning_rate': self.learning_rate,
                    'grad_global_norm': self.grad_global_norm,
                    'grad_norm_clip_val': self.grad_norm_clip_val,
                    'clipped_grad_global_norm': self.clipped_grad_global_norm,
                    'grad_clip_diff': self.clipped_grad_norm_diff})

    def _build_merged_tensorboard_summaries(self, scope, ops_dict,
        summary_fcn=tf.summary.scalar):
        ''' Builds and merges Tensorboard summaries.

        Args:
            scope: string for defining the scope the Tensorboard summaries to
            be created. This defines organizational structure within
            Tensorbaord.

            ops_dict: dictionary with string names as keys and TF objects as values. Names will be used as panel labels in Tensorboard.

            summary_fcn (optional): The Tensorflow summary function to be applied to TF objects in ops dict. Default: tf.summary_scalar

        Returns:
            A merged TF summary that, once executed via session.run(...), can be sent to Tensorboard via add_summary(...).
        '''

        summaries = []
        with tf.variable_scope(scope, reuse=False):
            for name, op in ops_dict.iteritems():
                summaries.append(summary_fcn(name, op))

        return tf.summary.merge(summaries)

    def _update_valid_tensorboard_summaries(self, valid_summary):
        '''Updates the Tensorboard summaries corresponding to the validation
        data. Only called if do_save_tensorboard_summaries.

        Args:
            valid_summary: dict returned by predict().

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _setup_tensorboard_images(self):
        '''Sets up Tensorboard Images. Called within first call to
        _update_tensorboard_images(). Requires the following have already been called:
            _maybe_setup_tensorboard(...)
            _maybe_setup_visualizations(...)

        Args:
            figs: dict with string figure names as keys and
            matplotlib.pyplot.figure objects as values.

        Returns:
            None.
        '''
        figs = self.figs

        if len(figs) == 0:
            # If no figs have been created, there's nothing to do here.
            return;

        if 'images' not in self.tensorboard:
            # The first time this function is called
            images = {
                'placeholders': dict(), # dict of tf placeholders
                'summaries': [], # list of tf.summary.images
                }
        else:
            '''
            In the event that this function is called multiple times,
            don't recreate existing image placeholders, but do create new ones
            if needed due to new figs having been created since last time.
            '''
            images = self.tensorboard['images']

        for fig_name, fig in figs.iteritems():

            if fig_name in images['placeholders']:
                # Don't recreate existing image placeholders
                continue

            (fig_width, fig_height) = fig.canvas.get_width_height()

            images['placeholders'][fig_name] = \
                tf.placeholder(tf.uint8, (1, fig_height, fig_width, 3))

            tb_fig_name = self._tensorboard_image_name(fig_name)

            images['summaries'].append(
                tf.summary.image(
                    tb_fig_name,
                    images['placeholders'][fig_name],
                    max_outputs=1))

        '''
        If this is a repeat call to the function, this will orphan an existing
        TF op :-(.
        '''
        images['merged_summaries'] = tf.summary.merge(images['summaries'])

        self.tensorboard['images'] = images

    def _update_tensorboard_images(self):
        ''' Imports figures into Tensorboard Images. Only called if:
                do_save_tensorboard_images and
                    (do_generate_training_visualizations or
                        do_generate_lvl_visualizations)

        Args:
            figs: dict with string figure names as keys and
            matplotlib.pyplot.figure objects as values.

        Returns:
            None.
        '''

        figs = self.figs

        if len(figs) == 0:
            # If no figs have been created, there's nothing to do here.
            return;

        # This done only on the first call to _update_tensorboard_images
        if 'images' not in self.tensorboard:
            self._setup_tensorboard_images()

        images = self.tensorboard['images']

        # Check to see whether any new figures have been added since the
        # tensorboard images were last setup. If so, efficiently redo that
        # setup. This orphans a TF op :-(. See _setup_tensorboard_images(...)
        for fig_name in figs:
            if fig_name not in images:
                self._setup_tensorboard_images()
                images = self.tensorboard['images']
                break

        # Convert figures into RGB arrays in a feed_dict for Tensorflow
        images_feed_dict = {}
        for fig_name in figs:
            key = images['placeholders'][fig_name]
            images_feed_dict[key] = self._fig2array(figs[fig_name])

        ev_merged_image_summaries = self.session.run(
            images['merged_summaries'], feed_dict=images_feed_dict)

        self.tensorboard['writer'].add_summary(
            ev_merged_image_summaries, self._step)

    @staticmethod
    def _tensorboard_image_name(fig_name):
        ''' Replaces all instances of '/' with '-'. Facilitates
        differentiating figure paths from Tensorboard Image scopes.

        Args:
            fig_name: string, e.g., 'partial/path/to/your/figure'

        Returns:
            Updated version fig_name, e.g., 'partial-path-to-your/figure'.
        '''

        key = os.sep
        replacement = '-'

        return fig_name.replace(key, replacement)

    @staticmethod
    def _fig2array(fig):
        ''' Convert from matplotlib figure to a numpy array suitable for a
        tensorboard image summary.

        Returns a numpy array of shape [1,width,height,3] (last for RGB)
        (Modified from plot_lfads.py).
        '''

        # This call is responsible for >95% of fig2array time
        fig.canvas.draw()

        # This call is responsible for 1%-5% of fig2array time
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

        # The following is responsible for basically 0% of fig2array time,
        # regardless of VERSION 1 vs VERSION 2.

        # VERSION 1
        # data_wxhx3 = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # data_1xwxhx3 = np.expand_dims(data_wxhx3,axis=0)

        # VERSION 2
        data_1xwxhx3 = data.reshape(
            (1,) + fig.canvas.get_width_height()[::-1] + (3,))

        return data_1xwxhx3

    # *************************************************************************
    # Training ****************************************************************
    # *************************************************************************

    def train(self, train_data=None, valid_data=None):
        '''Trains the model, managing the following core tasks:

            -randomly batching or generating training data
            -updating model parameters via gradients over each data batch
            -periodically evaluating the validation data
            -periodically updating visualization
            -periodically saving model checkpoints

        By convention, this call is the first time this object sees any data
        (train and valid, or as generated batch-by-batch by
        _get_data_batches(...)).

        Args:
            train_data (optional): dict containing the training data. If not
            provided (i.e., train_data=None), the subclass implementation of
            _get_data_batches(...) must generate training data on the fly.
            Default: None.

            valid_data (optional): dict containing the validation data.
            Default: None.

        Returns:
            None.
        '''

        N_EPOCH_SPLITS = 6 # Number of segments to time for profiling
        do_check_lvl = valid_data is not None

        self._setup_training(train_data, valid_data)

        if self._is_training_complete(self._ltl, do_check_lvl):
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

            if self._is_training_complete(epoch_loss, do_check_lvl):
                break

            epoch_timer.split('other')
            epoch_timer.disp()

        self._close_training(train_data, valid_data)

    def _setup_training(self, train_data, valid_data=None):
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

    def _train_epoch(self, data_batches):
        '''Performs training steps across an epoch of training data batches.

        Args:
            data_batches: list of dicts, where each dict contains a batch of
            training data.

        Returns:
            epoch_loss: float indicating the average loss across the epoch of
            training batches.
        '''
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

        epoch_loss = self._compute_epoch_average(batch_losses, batch_sizes)

        # Update lowest training loss (if applicable)
        if epoch_loss < self._ltl:
            self._update_ltl(epoch_loss)

        self.adaptive_learning_rate.update(epoch_loss)

        epoch_grad_norm = self._compute_epoch_average(
            batch_grad_norms, batch_sizes)
        self.adaptive_grad_norm_clip.update(epoch_grad_norm)

        self._increment_epoch()
        self._print_epoch_update(epoch_loss)

        # This should remain the final line before the return
        # (otherwise printing can provide misinformation, or worse)
        self.prev_loss = epoch_loss

        return epoch_loss

    def _print_epoch_update(self, epoch_loss):
        ''' Prints an update describing the optimization's progress, to be called at the end of each epoch.

        Args:
            epoch loss:

        Returns:
            None.
        '''

        if self.prev_loss is None:
            loss_improvement = np.nan
        else:
            loss_improvement = self.prev_loss - epoch_loss

        print('Epoch %d:' % self._epoch)
        print('\tTraining loss: %.2e;' % epoch_loss)
        print('\tImprovement in training loss: %.2e;' % loss_improvement)
        print('\tLearning rate: %.2e;' %  self.adaptive_learning_rate())
        print('\tLogging to: %s' % self.run_dir)

    def _train_batch(self, batch_data):
        '''Runs one training step. This function must evaluate the Tensorboard
        summaries:
            self.merged_opt_summary
            self.merged_hist_summary

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

    def predict(self, data, do_batch=False):
        ''' Runs a forward pass through the model using given input data. If
        the input data are larger than the batch size, the data are processed
        sequentially in multiple batches.

        Args:
            data: dict containing requisite data for generating predictions.
            Key/value pairs will be specific to the subclass implementation.

        Returns:
            predictions: dict containing model predictions based on data. Key/
            value pairs will be specific to the subclass implementation.

            summary: dict containing high-level summaries of the predictions.
            Key/value pairs will be specific to the subclass implementation.
            Must contain key: 'loss' whose value is a scalar indicating the
            evaluation of the overall objective function being minimized
            during training.

        If/when saving checkpoints, predictions and summary are saved into
        separate files. By placing lightweight objects as values in summary
        (e.g., scalars), the summary file can be loaded faster for post-
        training analyses that do not require loading the potentially bulky
        predictions.
        '''

        if do_batch:
            # THIS MESSES WITH VISUALIZATIONS. Here data are randomly batched,
            # then recombined. But no one outside of this function knows about
            # the effective shuffling of trials!
            #
            # To do: ensure trials are recombined into their original order.
            # This will require tracking trial IDs and passing those to
            # _combined_prediction_batches().

            batches_list, idx_list = self._split_data_into_batches(data)
            n_batches = len(batches_list)
            pred_list = []
            summary_list = []
            for batch_idx in range(n_batches):

                batch_data = batches_list[batch_idx]
                batch_size = self._get_batch_size(batch_data)

                print('\t\tPredict: batch %d of %d (%d trials)'
                      % (batch_idx+1, n_batches, batch_size))

                batch_predictions, batch_summary = self._predict_batch(
                    batch_data)

                pred_list.append(batch_predictions)
                summary_list.append(batch_summary)

            predictions, summary = self._combine_prediction_batches(
                pred_list, summary_list, idx_list)

        else:
            predictions, summary = self._predict_batch(data)

        assert ('loss' in summary),\
            ('summary must minimally contain key: \'loss\', but does not.')

        return predictions, summary

    def _predict_batch(self, batch_data):
        ''' Runs a forward pass through the model using a single batch of data.

        Args:
            data: dict containing requisite data for generating predictions.
            Key/value pairs will be specific to the subclass implementation.

        Returns:
            predictions: See docstring for predict().

            summary:  See docstring for predict().
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

    def _get_data_batches(self, data=None):
        ''' Splits data into batches OR generates data batches on the fly.

        Args:
            data (optional): a set of data examples to be randomly split into
            batches (e.g., using _split_data_into_batches(...)). Type can be
            subclass dependent (e.g., dict, list, custom class). If not
            provided, _generate_data_batches() will be used (and thus must be
            implemented in the subclass). Default: None.

        Returns:
            data_list: list of dicts, where each dict contains one batch of
            data.
            '''

        if data is None:
            return self._generate_data_batches()
        else:
            # Cleaner currently to not return idx_list, since otherwise would
            # require output argument handling in train().
            data_batches, idx_list = self._split_data_into_batches(data)
            return data_batches

    def _split_data_into_batches(self, data):
        ''' Randomly splits data into a set of batches. If the number of
        trials in data evenly divides by max_batch_size, all batches have size
        max_batch_size. Otherwise, the last batch is smaller containing the
        remainder trials.

        Args:
            data: dict containing the to-be-split data.

        Returns:
            data_list: list of dicts, where each dict contains one batch of
            data.

            idx_list: list, where each element, idx_list[i], is a list of the
            trial indices for the corresponding batch of data in data_list[i].
            This is used to recombine the trials back into their original
            (i.e., pre-batching) order by _combine_prediction_batches().
        '''

        n_trials = self._get_batch_size(data)
        max_batch_size = self.hps.max_batch_size
        n_batches = int(np.ceil(float(n_trials)/max_batch_size))

        shuffled_indices = range(n_trials)
        self.rng.shuffle(shuffled_indices)

        data_batches = []
        batch_indices = []

        start = 0
        for i in range(n_batches):

            stop = min(start + max_batch_size, n_trials)

            batch_idx = shuffled_indices[start:stop]
            batch_indices.append(batch_idx)
            batch_data = self._subselect_batch(data, batch_idx)
            data_batches.append(batch_data)

            start = stop

        return data_batches, batch_indices

    def _subselect_batch(self, data, batch_idx):
        ''' Subselect a batch of data given the batch indices.

        Args:
            data: dict containing the to-be-subselected data.

            batch_idx:  that specifies the subselection
            of the data (e.g., an array-like of trial indices).

        Returns:
            subselected_data: dict containing the subselected data.
        '''

        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _combine_prediction_batches(self, pred_list, summary_list, idx_list):
        ''' Combines predictions and summaries across multiple batches. This is
        required by predict(...), which first splits data into multiple
        batches (if necessary) before sequentially calling _predict_batch(...)
        on each data batch.

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

            idx_list: list, where each element is a list of trial indexes as
            returned by _split_data_into_batches(...). This is used to
            restore the original ordering of the trials (i.e., pre batching).
        '''

        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

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
            if np.mod(self._epoch, n) == 0:
                self.update_validation(train_data, valid_data)

    def update_validation(self, train_data, valid_data):
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

        if self.hps.do_save_tensorboard_summaries:
            self._update_valid_tensorboard_summaries(summary)

        self._maybe_save_lvl_checkpoint(
            summary['loss'], train_data, valid_data)

    @staticmethod
    def _compute_epoch_average(batch_vals, batch_sizes):
        '''Computes a weighted average of evaluations of a summary
        statistic across an epoch of data batches. This is all done in
        numpy (no Tensorflow).

        Args:
            batch_vals: list of floats containing the summary statistic
            evaluated on each batch.

            batch_sizes: list of ints containing the number of data
            examples per data batch.

        Returns:
            avg: float indicating the weighted average.
        '''
        weights = np.true_divide(batch_sizes, np.sum(batch_sizes))
        avg = np.dot(weights, batch_vals)
        return avg

    def _is_training_complete(self, loss, do_check_lvl=True):
        '''Determines whether the training optimization procedure should
        terminate. Termination criteria, governed by hyperparameters, are
        thresholds on the following:

            1) the training loss
            2) the learning rate
            3) the number of training epochs performed
            4) the number of training epochs performed since the lowest
               validation loss improved (only if do_check_lvl == True).

        Args:
            loss: float indicating the most recent loss evaluation across the
            training data.

        Returns:
            bool indicating whether any of the termination criteria have been
            met.
        '''
        hps = self.hps

        if self.is_done(self.run_dir):
            print('Stopping optimization: found .done file.')
            return True

        if loss is np.inf:
            print('\nStopping optimization: loss is Inf!')
            return True

        if np.isnan(loss):
            print('\nStopping optimization: loss is NaN!')
            return True

        if hps.min_loss is not None and loss <= hps.min_loss:
            print ('\nStopping optimization: loss meets convergence criteria.')
            return True

        if self.adaptive_learning_rate.is_finished(do_check_step=False):
            print ('\nStopping optimization: minimum learning rate reached.')
            return True

        if self.adaptive_learning_rate.is_finished(do_check_rate=False):
            print('\nStopping optimization:'
                  ' reached maximum number of training epochs.')
            return True

        if hps.max_train_time is not None and \
            self._get_train_time() > hps.max_train_time:

            print ('\nStopping optimization: training time exceeds '
                'maximum allowed.')
            return True

        if do_check_lvl:
            # Check whether lvl has been given a value (after being
            # initialized to np.inf), and if so, check whether that value has
            # improved recently.
            if not np.isinf(self._lvl) and \
                self._epoch - self.epoch_last_lvl_improvement >= \
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
            self._save_seso_checkpoint()

        # Save .done file. Critically placed after saving final checkpoint,
        # but before doing a bunch of other stuff that might fail. This way,
        # if anything below does fail, the .done file will be present,
        # indicating safe to interpret checkpoint model as final.
        self._save_done_file()

        if self.hps.do_generate_final_visualizations:

            self.update_visualizations(train_data, valid_data, is_final=True)

            if self.hps.do_save_tensorboard_images:
                self._update_tensorboard_images()

            if self.hps.do_save_final_visualizations:
                self.save_visualizations()

        if self.hps.do_generate_lvl_visualizations:
            if self.hps.do_save_lvl_ckpt:
                # Generate LVL visualizations
                print('\tGenerating visualizations from restored LVL model...')
                self.restore_from_lvl_checkpoint()
                self.update_visualizations(train_data, valid_data, is_lvl=True)

                if self.hps.do_save_tensorboard_images:
                    self._update_tensorboard_images()

                if self.hps.do_save_lvl_visualizations:
                    self.save_visualizations()
            else:
                raise Warning('Attempted to generate LVL visualizations, '
                    'but cannot because no LVL model checkpoint was saved.')

        if self.hps.do_log_output:
            self._log_file.close()

            # Redirect all printing, errors, and warnings back to defaults
            sys.stdout = self._default_stdout
            sys.stderr = self._default_stderr

    # *************************************************************************
    # Visualizations **********************************************************
    # *************************************************************************

    def _maybe_update_training_visualizations(self, train_data, valid_data):
        '''Updates visualizations if the current epoch number indicates that
        an update is due. Saves those visualization to Tensorboard or to
        individual figure file, depending on hyperparameters
        (do_save_tensorboard_images and do_save_training_visualizations,
        respectively.)

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''

        hps = self.hps
        if hps.do_generate_training_visualizations and \
            np.mod(self._epoch, hps.n_epochs_per_visualization_update) == 0:

            self.update_visualizations(train_data, valid_data, is_final=False)

            if hps.do_save_tensorboard_images:
                self._update_tensorboard_images()

            if hps.do_save_training_visualizations:
                self.save_visualizations()

    def update_visualizations(self, train_data, valid_data=None,
        is_final=False,
        is_lvl=False):
        '''Updates visualizations in self.figs. Only called if
            do_generate_training_visualizations OR
            do_generate_lvl_visualizations.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

            is_final: bool indicating if this call is made when the model is
            in its final state after training has terminated.

            is_lvl: bool indicating if this call is made when the model is in
            its lowest-validation-loss state.

            The two flags above can be used to signal generating a more
            comprehensive set of visualizations and / or analyses than those
            that are periodically generated throughout training.

        Returns:
            None.
        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def save_visualizations(self, format='eps', dpi=600):
        '''Saves individual figures to this run's figure directory. This is
        independent of Tensorboard Images.

        Args:
            format: (optional) string indicating the saved figure type (i.e.,
            file extension). See matplotlib.pyplot.figure.savefig(). Default:
            'eps'.

            dpi: (optional) dots per inch for saved figures. Default: 600.
        '''

        fig_dir = self.fig_dir
        for fig_name, fig in self.figs.iteritems():

            figs_dir_i, file_name_no_ext = os.path.split(
                os.path.join(fig_dir, fig_name))
            file_name = file_name_no_ext + '.' + format
            file_path = os.path.join(figs_dir_i, file_name)

            # This fig's dir may have additional directory structure beyond
            # the already existing .../figs/ directory. Make it.
            if not os.path.isdir(figs_dir_i):
                os.makedirs(figs_dir_i)

            fig.savefig(file_path, bbox_inches='tight', format=format, dpi=dpi)

    def _get_fig(self, fig_name,
        width=6.4,
        height=4.8,
        tight_layout=True):
        ''' Retrieves an existing figure or creates a new one.

        Args:
            fig_name: string containing a unique name for the requested
            figure. This is used to determine the filename to be used when
            saving the figure and the name of the corresponding Tensorboard
            Image. See also: tensorboard_image_name(...).

            width, height: (optional) width and height of requested figure, in
            inches. These are only used when creating a new figure--they does
            not update an existing one. Defaults: 6.4, 4.8.

            tight_layout (optional): See matplotlib.pyplot.figure docstring.

        Returns:
            The requested matplotlib.pyplot figure.
        '''

        if fig_name not in self.figs:
            self.figs[fig_name] = plt.figure(
                figsize=(width, height),
                tight_layout=tight_layout)

        fig = self.figs[fig_name]
        fig.clf()

        return fig

    @staticmethod
    def refresh_figs():
        ''' Refreshes all matplotlib figures.

        Args:
            None.

        Returns:
            None.
        '''
        if os.environ.get('DISPLAY','') == '':
            # If executing on a server with no graphical back-end
            pass
        else:
            plt.ion()
            plt.show()
            plt.pause(1e-10)

    # *************************************************************************
    # Scalar access and updates ***********************************************
    # *************************************************************************
    #
    # Convention: properties beginning with _ return numpy or python numeric
    # types (e.g., _epoch, _lvl). Their non-underscored counterparts
    # (e.g., epoch, lvl) are (or return) TF types, and most of these are setup
    # in _setup_records().

    @property
    def trainable_variables(self):
        ''' Returns a list of TF Variables that are updated during each call to
        _train_batch(...).

        Args: None

        Returns:
            A list of TF Variables.
        '''
        return tf.trainable_variables()

    @property
    def _epoch(self):
        '''Returns the number of training epochs taken thus far. An epoch is
        typically defined as one pass through all training examples, possibly
        using multiple batches (although this may depend on subclass-specific
        implementation details).

        Args:
            None.

        Returns:
            epoch: int specifying the current epoch number.
        '''
        return self.session.run(self.epoch)

    def _increment_epoch(self):
        self.session.run(self.increment_epoch)

    @property
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

    @property
    def _lvl(self):
        '''Returns the lowest validation loss encountered thus far during
        training.

        Args:
            None.

        Returns:
            lvl: float specifying the lowest validation loss.
        '''
        return self.session.run(self.lvl)

    def _update_lvl(self, lvl, epoch=None):
        ''' Updates the lowest validation loss and the epoch of this
        improvement.

        Args:
            lvl: A numpy scalar value indicating the (new) lowest validation
            loss.

            epoch (optional): Numpy scalar indicating the epoch of this
            improvement. Default: the current epoch.

        Returns:
            None.
        '''

        # Not Tensorflow
        if epoch is None:
            epoch = self._epoch

        # Not Tensorflow
        self.epoch_last_lvl_improvement = epoch

        # Tensorflow
        self.session.run(
            self.lvl_update, feed_dict={self.lvl_placeholder: lvl})

    @property
    def _ltl(self):
        '''Returns the lowest training loss encountered thus far (i.e., across
        an entire pass through the training data).

        Args:
            None.

        Returns:
            ltl: float specifying the lowest training loss.
        '''
        return self.session.run(self.ltl)

    def _update_ltl(self, ltl):
        ''' Updates the lowest training loss.

        Args:
            ltl: A numpy scalar value indicating the (new) lowest training
            loss.

        Returns:
            None.
        '''

        self.session.run(
            self.ltl_update,
            feed_dict={self.ltl_placeholder: ltl})

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

    # *************************************************************************
    # TF Variables access and updates *****************************************
    # *************************************************************************

    @property
    def n_params(self):
        ''' Counts the number of trainable parameters in a Tensorflow model
        (or scope within a model).

        Args:
            None

        Returns:
            integer specifying the number of trainable parameters.
        '''

        n_params = sum([np.prod(v.shape).value \
            for v in self.trainable_variables])

        return n_params

    def _get_vars_by_name_components(self, *name_components):
        ''' Returns TF variables whose names meet input search criteria.

        _get_vars_by_name_components(search_str1, search_str2, ...)

        Args:
            search_str1, search_str2, ... : strings to search for across all TF
            trainable variables. Variables will be returned only if they
            contain all of these strings.

        Returns:
            a list of TF variables whose name match the search criteria.
        '''
        matching_vars = []
        for v in self.trainable_variables:
            hits = [name_component in v.name
                for name_component in name_components]
            if all(hits):
                matching_vars.append(v)
        return matching_vars

    def update_variables_optimized(self, vars_to_train,
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

    def print_trainable_variables(self):
        '''Prints the current set of trainable variables.

        Args:
            None.

        Returns:
            None.
        '''
        print('\nTrainable variables:')
        for v in self.trainable_variables:
            print('\t' + v.name + ': ' + str(v.shape))
        print('')

    # *************************************************************************
    # Saving ******************************************************************
    # *************************************************************************

    def _maybe_save_checkpoint(self):
        '''Saves a model checkpoint if the current epoch number indicates that
        a checkpoint is due.

        Args:
            None.

        Returns:
            None.
        '''
        if self.hps.do_save_ckpt and \
            np.mod(self._epoch, self.hps.n_epochs_per_ckpt) == 0:

            self._save_seso_checkpoint()

    def _maybe_save_lvl_checkpoint(self, valid_loss, train_data, valid_data):
        '''Saves a model checkpoint if the current validation loss values is
        lower than all previously evaluated validation loss values. This
        includes saving model predictions over the training and validation
        data.

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
        if (self._epoch==0 or valid_loss < self._lvl):

            print('\t\tAchieved lowest validation loss.')

            self._update_lvl(valid_loss)

            if self.hps.do_save_lvl_ckpt:
                self._save_lvl_checkpoint()

            if self.hps.do_save_lvl_train_predictions or \
                self.hps.do_save_lvl_train_summaries:

                self.refresh_lvl_files(train_data, 'train')

            if self.hps.do_save_lvl_valid_predictions or \
                self.hps.do_save_lvl_valid_summaries:

                self.refresh_lvl_files(valid_data, 'valid')

    def _save_lvl_checkpoint(self):
        ''' Saves a lowest-validation-loss checkpoint.

        Args:
            None.

        Returns:
            None.
        '''
        print('\t\tSaving lvl checkpoint ...')
        self._save_checkpoint(self.savers['lvl'], self.lvl_ckpt_path)

    def _save_seso_checkpoint(self):
        ''' Saves an every-so-often checkpoint.

        Args:
            None.

        Returns:
            None.
        '''
        print('\tSaving checkpoint...')
        self._save_checkpoint(self.savers['seso'], self.ckpt_path)

    def _save_checkpoint(self, saver, ckpt_path):
        '''Saves a model checkpoint.

        Args:
            saver: Tensorflow saver to use, generated via tf.train.Saver(...).

            ckpt_path: string containing the path of the checkpoint to be
            saved.

        Returns:
            None.
        '''
        self._update_train_time()
        saver.save(self.session, ckpt_path, global_step=self._step)

        ckpt_dir, ckpt_fname = os.path.split(ckpt_path)
        self.adaptive_learning_rate.save(ckpt_dir)
        self.adaptive_grad_norm_clip.save(ckpt_dir)

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
                self._save_lvl_helper(predictions, filename_no_extension)

    def _maybe_save_lvl_summaries(self,
        summary, train_or_valid_str):

        if summary is not None:
            print('\t\tSaving lvl summary (%s).' % train_or_valid_str)
            # E.g., train_predictions or valid_summary
            filename_no_extension = train_or_valid_str + '_summary'
            self._save_lvl_helper(summary, filename_no_extension)

    def refresh_lvl_files(self, data, train_or_valid_str):
        '''Saves model predictions over the training or validation data.

        If prediction summaries are generated, those summaries are saved in
        separate .pkl files (and optional .mat files). See docstring for
        predict() for additional detail.

        Args:
            data: dict containing either the training or validation data.

            train_or_valid_str: either 'train' or 'valid', indicating whether
            data contains training data or validation data, respectively.

        Returns:
            None.
        '''

        if data is not None:
            pred, summary = self.predict(data)
            self._maybe_save_lvl_predictions(pred, train_or_valid_str)
            self._maybe_save_lvl_summaries(summary, train_or_valid_str)

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

    def _save_lvl_helper(self, data_to_save, filename_no_extension):
        '''Pickle and save data as .pkl file. Optionally also save the data as
        a .mat file.

            Args:
                data_to_save: any pickle-able object to be pickled and saved.

            Returns:
                None.
        '''

        pkl_path = os.path.join(self.lvl_dir, filename_no_extension)
        self._save_pkl(data_to_save, pkl_path)

        if self.hps.do_save_lvl_mat_files:
            mat_path = os.path.join(self.lvl_dir, filename_no_extension)
            self._save_mat(data_to_save, pkl_path)

    @staticmethod
    def _save_pkl(data_to_save, save_path_no_extension):
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

    @staticmethod
    def _save_mat(data_to_save, save_path_no_extension):
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

    # *************************************************************************
    # Loading and Restoring ***************************************************
    # *************************************************************************

    @staticmethod
    def exists_lvl_train_predictions(run_dir):
        return RecurrentWhisperer._exists_lvl(run_dir, 'train', 'predictions')

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
    def exists_lvl_train_summary(run_dir):
        return RecurrentWhisperer._exists_lvl(run_dir, 'train', 'summary')

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
    def exists_lvl_valid_predictions(run_dir):
        return RecurrentWhisperer._exists_lvl(run_dir, 'valid', 'predictions')

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
    def exists_lvl_valid_summary(run_dir):
        return RecurrentWhisperer._exists_lvl(run_dir, 'valid', 'summary')

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

    @classmethod
    def load_lvl_model(cls, run_dir, new_base_path=None):
        ''' Load an LVL model given only the run directory, properly handling
        subclassing.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            The desired model with restored LVL parameters.
        '''

        hps_dict = cls.load_hyperparameters(run_dir)

        if new_base_path is not None:
            # Handle loading a model that was saved on another system.
            # This functionality is still under development.
            raise NotImplementedError()

            local_log_dir = cls.update_dir_to_local_system(
                hps_dict['log_dir'], new_base_path)
            hps_dict['log_dir'] = local_log_dir

        hps_dict['do_custom_restore'] = True
        hps_dict['do_log_output'] = False
        model = cls(**hps_dict)

        lvl_ckpt = tf.train.get_checkpoint_state(model.lvl_dir)
        lvl_ckpt_path = lvl_ckpt.model_checkpoint_path

        if new_base_path is not None:
            lvl_ckpt_path = model.update_dir_to_local_system(
                lvl_ckpt_path, new_base_path)

        model.restore_from_lvl_checkpoint(model_checkpoint_path=lvl_ckpt_path)

        return model

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

    def _restore(self, saver, ckpt_dir, model_checkpoint_path):
        '''
        COMMENTS NEED UPDATING
        '''
        saver.restore(self.session, model_checkpoint_path)
        self.adaptive_learning_rate.restore(ckpt_dir)
        self.adaptive_grad_norm_clip.restore(ckpt_dir)

        # Resume training timer from value at last save.
        self.train_time_offset = self.session.run(self.train_time)

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
        self._restore(self.savers['lvl'], self.lvl_dir, model_checkpoint_path)

    def _restore_from_checkpoint(self, saver, ckpt_dir):
        '''Restores a model from the most advanced previously saved model
        checkpoint.

        Note that the hyperparameters files are not updated from their original
        form. This can become relevant if restoring a model and resuming
        training using updated values of non-hash hyperparameters.

        Args:
            saver: Tensorflow saver to use, generated via tf.train.Saver(...).

            ckpt_dir: string containing the path to the directory containing
            the checkpoint to be restored.

        Returns:
            None.
        '''
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if not(tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
            raise FileNotFoundError('Checkpoint does not exist: %s'
                                    % ckpt.model_checkpoint_path)

        # Restore previous session
        print('Previous checkpoints found.')
        print('Loading latest checkpoint: %s.'
              % ntpath.basename(ckpt.model_checkpoint_path))
        self._restore(
            saver, ckpt_dir, ckpt.model_checkpoint_path)

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

        return RecurrentWhisperer._load_pkl(path_to_file)

    @staticmethod
    def _exists_lvl(
        run_dir,
        train_or_valid_str,
        predictions_or_summary_str):
        '''Checks if previously saved model predictions or summaries exist.

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
            True if the lvl file exists.
        '''

        path_to_file = RecurrentWhisperer._get_lvl_path(
            run_dir, train_or_valid_str, predictions_or_summary_str)

        return os.path.exists(path_to_file)

    @staticmethod
    def _load_pkl(path_to_file):
        '''Loads previously saved data.

        Args:
            path_to_file: string containing the path to the saved .pkl data.

        Returns:
            dict containing saved data.
        '''
        if os.path.exists(path_to_file):
            file = open(path_to_file, 'rb')
            load_path = file.read()
            data = cPickle.loads(load_path)
            file.close()
        else:
            raise IOError('%s not found.' % path_to_file)

        return data
