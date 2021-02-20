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
import logging
from copy import deepcopy
from subprocess import call
import pdb

import tensorflow as tf
import numpy as np
import numpy.random as npr

# Imports for saving data, predictions, summaries
import cPickle
import h5py, json
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
    functions (see docstrings in the corresponding function prototypes
    throughout this file):

    _default_hash_hyperparameters()
    _default_non_hash_hyperparameters()
    _setup_model(...)
    _train_batch(...)
    _predict_batch(...)
    _get_batch_size(...)
    _subselect_batch(...)

    Required only if generating (or augmenting) data on-the-fly during
    training:
    generate_data(...)

    Required only if do_batch_predictions:
    _combine_prediction_batches(...)

    Not required, but can provide additional helpful functionality:
    _setup_training(...)
    _update_valid_tensorboard_summaries
    _update_visualizations(...)
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
            in the latter category do not affect this hash so that one can
            more readily interact with a training run without retraining a
            model from scratch (e.g., change printing or visualization
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

                do_batch_predictions: bool indicating whether to compute
                predictions in batches, or all in a single evaluation of the
                TF graph. This may be required for GPU/CPU memory management
                in large models or for large datasets. Default: False.

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

                do_save_seso_ckpt: bool indicating whether or not to save model
                checkpoints. SESO = save-every-so-often. Default: True.

                do_save_ltl_ckpt: bool indicating whether or not to save model
                checkpoints specifically when a new lowest training loss is
                achieved. Default: True.

                do_save_lvl_ckpt: bool indicating whether or not to save model
                checkpoints specifically when a new lowest validation loss is
                achieved. Default: True.

                fig_filetype: string indicating the saved figure type (i.e.,
                file extension). See matplotlib.pyplot.figure.savefig().
                Default: 'pdf'.

                fig_dpi: dots per inch for saved figures. Default: 600.

                predictions_filetype: string indicating the filetype for
                saving model predictions. Options include 'npz', 'h5', 'mat',
                'pkl', 'json'. Default: 'npz'.

                summary_filetype: string indicating the filetype for saving
                prediction summaries. Options include 'npz', 'h5', 'mat',
                'pkl', 'json'. Default: 'npz'.

                ***************************************************************
                WHEN, AND HOW OFTEN, TO GENERATE AND SAVE VISUALIZATIONS ******
                ***************************************************************

                do_generate_pretraining_visualizations: bool indicating
                whether or not to generate visualizations using the
                initialized (untrained) model. Beyond the diagnostic value,
                this can be helpful in forcing errors early if there are bugs
                in your visualizations code (rather than after an initial
                round of training epochs). Default: False.

                do_save_pretraining_visualizations: bool indicating whether or
                not to save individual figure files for the pre-training
                visualizations. Only relevant if
                do_generate_pretraining_visualizations. Default: False.

                do_generate_training_visualizations: bool indicating whether or
                not to generate visualizations periodically throughout
                training. Frequency is controlled by
                n_epoochs_per_visualization_update. Default: True.

                do_save_training_visualizations: bool indicating whether or not
                to save individual figure files as they are generated
                periodically throughout training. Only relevant if
                do_generate_training_visualizations. Default: True.

                do_generate_final_visualizations: bool indicating whether or
                not to generate visualizations using the final state of the
                model (i.e., upon termination of training). Default: True.

                do_save_final_visualizations: bool indicating whether or not
                to save individual figure files for the final visualizations.
                Only relevant if do_generate_final_visualizations.
                Default: True.

                do_generate_ltl_visualizations: bool indicating whether or not
                to, after training is complete, load the LTL model and generate
                visualization from it. Default: True.

                do_save_ltl_visualizations: bool indicating whether or not to
                save individual figure files for the LTL visualizations. Only
                relevant if do_generate_ltl_visualizations Default: True.

                do_save_ltl_train_summary: bool indicating whether to
                save prediction summaries over the training data each time the
                model achieves a new lowest training loss. Default: True.

                do_save_ltl_train_predictions: bool indicating whether to,
                after training is complete, load the LTL model, generate
                predictions over the training data, and save those predictions
                to disk. Note, because this can be time consuming, this is
                only done once, rather than continually throughout training.
                Default: True.

                do_generate_lvl_visualizations: bool indicating whether or not
                to, after training is complete, load the LVL model and generate
                visualization from it. Default: True.

                do_save_lvl_visualizations: bool indicating whether or not to
                save individual figure files for the LVL visualizations. Only
                relevant if do_generate_lvl_visualizations. Default: True.

                do_save_lvl_train_predictions: bool indicating whether to
                maintain a .pkl file containing predictions over the training
                data based on the lowest-validation-loss parameters.

                do_save_lvl_train_summary: bool indicating whether to
                maintain a .pkl file containing summaries of the training
                predictions based on the lowest-validation-loss parameters.

                do_save_lvl_valid_predictions: bool indicating whether to
                maintain a .pkl file containing predictions over the validation
                data based on the lowest-validation-loss parameters.

                do_save_lvl_valid_summary: bool indicating whether to
                maintain a .pkl file containing summaries of the validation
                 predictions based on the lowest-validation-loss parameters.

                max_seso_ckpt_to_keep: int specifying the maximum number of
                save-every-so-often model checkpoints to keep around.
                Default: 1.

                max_ltl_ckpt_to_keep: int specifying the maximum number
                of lowest-training-loss (ltl) checkpoints to maintain.
                Default: 1.

                max_lvl_ckpt_to_keep: int specifying the maximum number
                of lowest-validation-loss (lvl) checkpoints to maintain.
                Default: 1.

                n_epochs_per_seso_update: int specifying the number of epochs
                between save-every-so-often (seso) checkpoint saves.
                Default: 100.

                n_epochs_per_lvl_update: int specifying the number of
                epochs between evaluating predictions over the validation
                data. Default: 100.

                n_epochs_per_visualization_update: int specifying the number
                of epochs between updates of any visualizations. Default: 100.

                device_type: Either 'cpu' or 'gpu', indicating the type of
                hardware device that will support this model. Default: 'gpu'.

                device_id: Nonnegative integer specifying the CPU core ID
                (for device_type: 'cpu') or GPU ID (for device_type: 'gpu') of
                the specific local hardware device to be used for this model.

                cpu_device_id: Nonnegative integer specifying the ID of the
                CPU core to be used for CPU-only operations. Default: 0.

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
        self.epoch_loss = None
        self.adaptive_learning_rate = AdaptiveLearningRate(**hps.alr_hps)
        self.adaptive_grad_norm_clip = AdaptiveGradNormClip(**hps.agnc_hps)

        self._setup_run_dir()

        self.timer = Timer(
            name='Total run time',
            do_retrospective=True)
        self.timer.start()

        self._setup_devices()

        with tf.variable_scope(hps.name, reuse=tf.AUTO_REUSE):

            with tf.device(self.cpu_device):
                self._setup_records()
                self.timer.split('_setup_records')

            with tf.device(self.device):
                self._setup_model()
                self.timer.split('_setup_model')

                self._setup_optimizer()
                self.timer.split('_setup_optimizer')

                self._setup_visualizations()
                self.timer.split('_setup_visualizations')

                # Each of these will create run_dir if it doesn't exist
                # (do not move above the os.path.isdir check that is in
                # _setup_run_dir)
                self._maybe_setup_tensorboard_summaries()
                self.timer.split('_setup_tensorboard')

                self._setup_savers()
                self.timer.split('_setup_savers')

                self._setup_session()
                self.timer.split('_setup_session')

                if not hps.do_custom_restore:
                    self.initialize_or_restore()
                    self.print_trainable_variables()
                    self.timer.split('_initialize_or_restore')

    # *************************************************************************
    # Hyperparameters management **********************************************
    # *************************************************************************

    @classmethod
    def default_hyperparameters(cls):
        ''' Returns the dict of ALL (RecurrentWhisperer + subclass)
        hyperparameters (both hash and non-hash). This is needed for
        command-line argument parsing.

        Args:
            None.

        Returns:
            dict of hyperparameters.

        '''

        hps = cls.default_hash_hyperparameters()
        non_hash_hps = cls.default_non_hash_hyperparameters()

        hps.update(non_hash_hps)

        return hps

    @classmethod
    def default_hash_hyperparameters(cls):
        ''' Returns the dict of ALL (RecurrentWhisperer + subclass)
        hyperparameters that are included in the run hash.

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''

        hash_hps = Hyperparameters.integrate_hps(
            RecurrentWhisperer._default_rw_hash_hyperparameters(),
            cls._default_hash_hyperparameters())

        return hash_hps

    @classmethod
    def default_non_hash_hyperparameters(cls):
        ''' Returns the dict of ALL (RecurrentWhisperer + subclass)
        hyperparameters that are NOT included in the run hash.

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''

        non_hash_hps = Hyperparameters.integrate_hps(
            RecurrentWhisperer._default_rw_non_hash_hyperparameters(),
            cls._default_non_hash_hyperparameters())
        return non_hash_hps

    @staticmethod
    def _default_rw_hash_hyperparameters():
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
    def _default_rw_non_hash_hyperparameters():
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

            'log_dir': '/tmp/rnn_logs/',
            'run_script': None,
            'n_folds': None,
            'fold_idx': None,

            # Termination criteria
            'min_loss': None,
            'max_train_time': None,
            'max_n_epochs_without_ltl_improvement': 200,
            'max_n_epochs_without_lvl_improvement': 200,

            'do_batch_predictions': False,

            'do_log_output': False,
            'do_restart_run': False,
            'do_custom_restore': False,

            # Tensorboard logging
            'do_save_tensorboard_summaries': True,
			'do_save_tensorboard_histograms': True,
            'do_save_tensorboard_images': True,

            # Frequency of (potentially time consuming) operations
            'n_epochs_per_seso_update': 100,
            'n_epochs_per_ltl_update': 100,
            'n_epochs_per_lvl_update': 100,
            'n_epochs_per_visualization_update': 100,

            # Save-every-so-often Visualizations
            'do_generate_pretraining_visualizations': False,  # (pre-training)
            'do_save_pretraining_visualizations': False,

            # These correspond with n_epochs_per_visualization_update
            'do_generate_training_visualizations': True,      # (peri-training)
            'do_save_training_visualizations': True,

            'do_generate_final_visualizations': True,         # (post-training)
            'do_save_final_visualizations': True,


            # Save-every-so-often (seso) checkpoints
            'do_save_seso_ckpt': True,
            'max_seso_ckpt_to_keep': 1,

            # Lowest-training-loss (LTL):
            # Checkpoint and prediction summary are saved as often as every
            # n_epochs_per_ltl_update. Predictions and visualizations are saved
            # only once at the end of training upon restoring the LTL model.
            'do_save_ltl_ckpt': True,
            'do_save_ltl_train_summary': True,
            'do_save_ltl_train_predictions': True,
            'do_generate_ltl_visualizations': True,
            'do_save_ltl_visualizations': True,
            'max_ltl_ckpt_to_keep': 1,

            # Lowest-validation-loss (LVL) checkpoints
            # Only relevant if valid_data is provided to train(...).
            # Checkpoint and summary are saved as often as every
            # n_epochs_per_lvl_update. Predictions and visualizations are saved
            # only once at the end of training upon restoring the LVL model.
            'do_save_lvl_ckpt': True,
            'do_save_lvl_train_predictions': True,
            'do_save_lvl_train_summary': True,
            'do_save_lvl_valid_predictions': True,
            'do_save_lvl_valid_summary': True,
            'do_generate_lvl_visualizations': True,
            'do_save_lvl_visualizations': True,
            'max_lvl_ckpt_to_keep': 1,

            'fig_filetype': 'pdf',
            'fig_dpi': 600,
            'do_print_visualizations_timing': False,

            'predictions_filetype': 'npz',
            'summary_filetype': 'npz',

            # GPU / CPU device management
            'device_type': 'gpu',
            'device_id': 0,
            'cpu_device_id': 0,
            'per_process_gpu_memory_fraction': 1.0,
            'disable_gpus': False,
            'allow_gpu_growth': True,
            'allow_soft_placement': True,
            'log_device_placement': False,
        }

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
            cls.default_hash_hyperparameters(),
            cls.default_non_hash_hyperparameters())

    @staticmethod
    def get_command_line_call(run_script,
                              hp_dict={},
                              do_shell_format=False,
                              shell_delimiter=' \\\n'):

        ''' Generates a command line call to a user-specified shell script with
        RecurrentWhisperer hyperparameters passed in as command-line arguments.
        Can be formatted for execution within Python or from a shell script.

        Args:
            run_script: string specifying the shell script call,
            e.g., 'location/of/your/run_script.sh'

            hp_dict: (optional) dict containing any hps to override defaults.
            Default: {}

            do_format_for_shell: (optional) bool indicating whether to return
            the command-line call as a string (for writing into a higher-level
            shell script; for copying into a terminal). Default: False (see
            below).

        Returns:
            Default:
                cmd_list: a list that is interpretable by subprocess.call:
                    subprocess.call(cmd_list)

            do_shell_format == True:
                cmd_str: a string (suitable for placing in a shell script or
                copying into a terminal .
        '''

        def raise_error():
        	# This should not be reachable--Hyperparameters.flatten converts to
        	# a colon delimited format.
        	raise ValueError('HPs that are themselves dicts are not supported')

        flat_hps = Hyperparameters.flatten(hp_dict)
        hp_names = flat_hps.keys()
        hp_names.sort()

        if do_shell_format:

            cmd_str = 'python %s' % run_script
            for hp_name in hp_names:
                val = flat_hps[hp_name]
                if isinstance(val, dict):
                    omit_dict_hp(hp_name)
                else:
                    cmd_str += str(
                        '%s--%s=%s' % (shell_delimiter, hp_name, str(val)))

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

    @classmethod
    def execute_command_line_call(cls, run_script, hp_dict={}):
        ''' Executes a command line call to a user-specified shell script with
        RecurrentWhisperer hyperparameters passed in as command-line arguments.

        Args:
            run_script: string specifying the shell script call,
            e.g., 'location/of/your/run_script.sh'

            hp_dict: (optional) dict containing any hps to override defaults.
            Default: {}

        Returns:
            None.
        '''

        cmd_list = cls.get_command_line_call(run_script, hp_dict)
        print(cmd_list)
        call(cmd_list)

    @classmethod
    def write_shell_script(cls, save_path, run_script, hp_dict):
        file = open(save_path, 'w')
        shell_str = cls.get_command_line_call(
            run_script, hp_dict, do_shell_format=True)
        file.write(shell_str)
        file.close()

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
        subdirs, paths = self._build_paths(run_dir)

        self._subdirs = subdirs
        self._paths = paths
        self._run_hash = run_hash
        self._run_dir = run_dir

        hps_dir = subdirs['hps']
        seso_dir = subdirs['seso']
        ltl_dir = subdirs['ltl']
        lvl_dir = subdirs['lvl']

        if os.path.isdir(run_dir):
            print('\nRun directory found: %s.' % run_dir)
            ckpt = tf.train.get_checkpoint_state(seso_dir)
            ltl_ckpt = tf.train.get_checkpoint_state(ltl_dir)
            lvl_ckpt = tf.train.get_checkpoint_state(lvl_dir)
            if ckpt is None and ltl_ckpt is None and lvl_ckpt is None:
                print('No checkpoints found.')
            if self.hps.do_restart_run:
                print('\tDeleting run directory.')
                shutil.rmtree(run_dir)

                # Avoids pathological behavior whereby it is impossible to
                # restore a run that was started with do_restart_run = True.
                self.hps.do_restart_run = False

        if not os.path.isdir(run_dir):
            print('\nCreating run directory: %s.' % run_dir)

            # Subdirectories
            for d in subdirs.values():
                os.makedirs(d)

            # Sub-subdirectories
            for version in ['seso', 'ltl', 'lvl']:
                d = self._build_fig_dir(run_dir, version=version)
                os.makedirs(d)

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

        model_log_path = self._paths['model_log_path']
        loggers_log_path = self._paths['loggers_log_path']

        fh = logging.FileHandler(loggers_log_path)
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
        self._log_file = open(model_log_path, 'a+')
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

        ops = {} # Each value is a tf.Variable
        placeholders = {}
        update_ops ={}
        increment_ops = {}

        with tf.variable_scope('records', reuse=False):
            '''Maintain state using TF framework for seamless saving and
            restoring of runs'''

            # These are all begging for a simple class to reduce the code
            # copying.

            ''' Counter to track the current training epoch number. An epoch
            is defined as one complete pass through the training data (i.e.,
            multiple batches).'''
            ops['epoch'] = tf.Variable(0,
                name='epoch',
                trainable=False,
                dtype=tf.int32)
            increment_ops['epoch'] = tf.assign_add(ops['epoch'], 1,
                name='increment_epoch')

            ''' Timing TF variable to maintain timing information across
            potentially multiple training sessions. Allows for previous
            training time to be recalled upon restoring an existing model. '''
            ops['train_time'] = tf.Variable(0,
                name='train_time',
                trainable=False,
                dtype=self.dtype)
            placeholders['train_time'] = tf.placeholder(self.dtype,
                name='train_time')
            update_ops['train_time'] = tf.assign(
                ops['train_time'], placeholders['train_time'],
                name='update_train_time')

            ops['global_step'] = tf.Variable(0,
                name='global_step',
                trainable=False,
                dtype=tf.int32)

            # lowest validation loss
            (ops['lvl'],
            placeholders['lvl'],
            update_ops['lvl'],
            ops['epoch_last_lvl_improvement'],
            placeholders['epoch_last_lvl_improvement'],
            update_ops['epoch_last_lvl_improvement']) = \
                self._setup_loss_records('lvl')

            # lowest training loss
            (ops['ltl'],
            placeholders['ltl'],
            update_ops['ltl'],
            ops['epoch_last_ltl_improvement'],
            placeholders['epoch_last_ltl_improvement'],
            update_ops['epoch_last_ltl_improvement']) = \
                self._setup_loss_records('ltl')

        self.records = {
            'ops': ops,
            'placeholders': placeholders,
            'update_ops': update_ops,
            'increment_ops': increment_ops
            }

    def _setup_loss_records(self, version):
        ''' Helper function for building auxilliary TF data for maintaining
        state about loss values and history.

        Args:
            version: 'ltl' or 'lvl'.

        Returns:
            A lot.
        '''
        self._assert_version_is_ltl_or_lvl(version)

        op = tf.Variable(
            np.inf, name=version, trainable=False, dtype=self.dtype)

        ph = tf.placeholder(self.dtype, name=version)

        update_op = tf.assign(op, ph, name='update_%s' % version)

        epoch_last_improvement = tf.Variable(0,
            name='epoch_last_%s_improvement' % version,
            trainable=False,
            dtype=tf.int32)

        epoch_ph = tf.placeholder(
            tf.int32, name='epoch_last_%s_improvement' % version)

        update_epoch = tf.assign(epoch_last_improvement, epoch_ph,
            name='update_epoch_last_%s_improvement' % version)

        return (op, ph, update_op,
            epoch_last_improvement, epoch_ph, update_epoch)

    def _setup_devices(self):
        ''' Select the hardware devices to use for this model.

        This creates attributes:
            self.device, e.g., : 'gpu:0'
            self.cpu_device, e.g., : 'cpu:0'

        Args:
            None.

        Returns:
            None.
        '''

        device_type = self.hps.device_type
        device_id = self.hps.device_id

        assert device_type in ['cpu', 'gpu'], \
            'Unsupported device_type: %s' % str(device_type)

        if device_type == 'gpu':
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                cuda_devices = ''
            print('\n\nCUDA_VISIBLE_DEVICES: %s' % cuda_devices)

        self.device = '%s:%d' % (device_type, self.hps.device_id)
        print('Attempting to build TF model on %s\n' % self.device)

        ''' Some simple ops (e.g., tf.assign, tf.assign_add) must be placed on
        a CPU device. Instructing otherwise just ellicits warnings and
        overrides (at least in TF<=1.15).'''
        self.cpu_device = 'cpu:%d' % self.hps.cpu_device_id
        print('Placing CPU-only ops on %s\n' % self.cpu_device)

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
                zipped_grads, global_step=self.records['ops']['global_step'])

    def _setup_visualizations(self):
        '''Sets up visualizations. Only called if
            do_generate_training_visualizations or
            do_generate_lvl_visualizations.

        Args:
            None.

        Returns:
            figs: dict with string figure names as keys and
            matplotlib.pyplot.figure objects as values. Typical usage will
            populate this dict upon the first call to update_visualizations().
        '''
        self.figs = dict()

        # This timer is rebuilt each time visualizations are generated,
        # but is required here in case visualization functions are
        # called manually.
        self._setup_visualizations_timer()

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
            tf.global_variables(), max_to_keep=self.hps.max_seso_ckpt_to_keep)

        # lowest training loss
        self.savers['ltl'] = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.hps.max_ltl_ckpt_to_keep)

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

        config.allow_soft_placement = hps.allow_soft_placement
        config.log_device_placement = hps.log_device_placement

        if hps.per_process_gpu_memory_fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = \
                hps.per_process_gpu_memory_fraction

        self.session = tf.Session(config=config)
        print('\n')

    # *************************************************************************
    # Initializations *********************************************************
    # *************************************************************************

    def initialize_or_restore(self, version_priority=['seso', 'ltl', 'lvl']):
        '''Initializes all Tensorflow objects, either from an existing model
        checkpoint if detected or otherwise as specified in _setup_model. If
        starting a training run from scratch, writes a yaml file containing
        all hyperparameter settings.

        Args:
            version_priority (optional): list of checkpoint version strings
            arranged in order of preference for use when restoring. The first
            version found will be used. Default: ['seso', 'ltl', 'lvl'].

        Returns:
            None.
        '''

        for version in version_priority:
            if self.exists_checkpoint(version):
                self.restore_from_checkpoint(version)
                return

        hps_path = self._paths['hps_path']
        hps_yaml_path = self._paths['hps_yaml_path']
        run_script_path = self._paths['run_script_path']

        # Initialize new session
        print('Initializing new run (%s).' % self.hps.hash)
        self.session.run(tf.global_variables_initializer())

        self.hps.save_yaml(hps_yaml_path) # For visual inspection
        self.hps.save(hps_path) # For restoring a run via its run_dir
        # (i.e., without needing to manually specify hps)

        if self.hps.run_script is not None:
            self.write_shell_script(
                run_script_path,
                self.hps.run_script,
                self.hps.integrated_hps)

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
                    'lvl': self.records['ops']['lvl'],
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
        pass

    def _setup_tensorboard_images(self):
        '''Sets up Tensorboard Images. Called within first call to
        _update_tensorboard_images(). Requires the following have already been
        called:
            _maybe_setup_tensorboard(...)
            _maybe_setup_visualizations(...)

        Args:
            figs: dict with string figure names as keys and
            matplotlib.pyplot.figure objects as values.

        Returns:
            None.
        '''
        hps = self.hps
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
            tb_fig_name = self._tensorboard_image_name(fig_name)

            # Don't use precious GPU memory for these images, which are just
            # used for storage--they aren't computed on.
            with tf.device(self.cpu_device):

                images['placeholders'][fig_name] = tf.placeholder(
                    tf.uint8, (1, fig_height, fig_width, 3))

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

        self._visualizations_timer.split('Tensorboard setup')

        figs = self.figs

        if len(figs) == 0:
            # If no figs have been created, there's nothing to do here.
            return;

        # This done only on the first call to _update_tensorboard_images
        if 'images' not in self.tensorboard:
            self._setup_tensorboard_images()

        images = self.tensorboard['images']

        self._visualizations_timer.split('Images setup')

        # Check to see whether any new figures have been added since the
        # tensorboard images were last setup. If so, efficiently redo that
        # setup. This orphans a TF op :-(. See _setup_tensorboard_images(...)
        for fig_name in figs:
            if fig_name not in images:
                self._setup_tensorboard_images()
                images = self.tensorboard['images']
                break

        self._visualizations_timer.split('Building RGB arrays')

        # Convert figures into RGB arrays in a feed_dict for Tensorflow
        images_feed_dict = {}
        for fig_name in figs:
            key = images['placeholders'][fig_name]
            images_feed_dict[key] = self._fig2array(figs[fig_name])

        self._visualizations_timer.split('Graph Ops')

        ev_merged_image_summaries = self.session.run(
            images['merged_summaries'], feed_dict=images_feed_dict)

        self.tensorboard['writer'].add_summary(
            ev_merged_image_summaries, self._step)

        self._visualizations_timer.split('Transition from Tensorboard')

    def _tensorboard_image_name(self, fig_name):
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

        There are three supported modes of operation:
        1. Generate on-the-fly training data (new data for each gradient step).
           Here, call train(train_data=None, valid_data=None), i.e., train().
           You must provide an implementation of generate_data().
        2. Provide a single, fixed set of training data. This is done by
           calling train(train_data).
        3. Provide, single, fixed set of training data (as in 2) and a single,
           fixed set of validation data. This is done by calling
           train(train_data, valid_data).

        The specific mode invoked determines, along with user options in
        self.hps, whether SESO, LTL, and/or LVL updates are generated and
        saved throughout training. Each update can optionally save a model
        checkpoint, predictions, prediction summaries, and/or visualization.
        All modes support save-every-so-often (SESO) and lowest-training-loss
        (LTL) updates. Only mode 3 supports the lowest-validation-loss (LVL)
        updates.

        By convention, this call is the first time this object sees any data
        (train_data / valid_data, or as generated batch-by-batch by
        generate_data()).

        Args:
            train_data (optional): dict containing the training data. If not
            provided (i.e., train_data=None), the subclass implementation of
            generate_data(...) must generate training data on the fly.
            Default: None.

            valid_data (optional): dict containing the validation data.
            Default: None.

        Returns:
            None.
        '''
        self._setup_training(train_data, valid_data)
        self.timer.split('_setup_training')

        if self._is_training_complete(self._ltl):
            # If restoring from a completed run, do not enter training loop
            # and do not save a new checkpoint.
            return

        self._maybe_update_visualizations_pretraining(train_data, valid_data)
        self.timer.split('_init_visualizations')

        # Training loop
        print('Entering training loop.')
        done = False

        while not done:

            self._initialize_epoch()

            # Profiling for data prep is managed elsewhere.
            # It's complicated in the case of on-the-fly data generation.
            epoch_train_data = self._prepare_epoch_data(train_data)

            train_pred, train_summary = self._train_epoch(epoch_train_data)
            self._epoch_timer.split('train')

            self._maybe_save_seso_checkpoint()
            self._epoch_timer.split('seso')

            self._maybe_save_ltl_checkpoint(train_pred, train_summary)
            self._epoch_timer.split('ltl')

            if self._do_predict_validation:
                # ...if using validation data and due for an update this epoch
                valid_pred, valid_summary = self.predict(
                    valid_data, train_or_valid_str='valid')

                # ... if validation loss is better than previously seen
                self._maybe_save_lvl_checkpoint(
                    train_pred=train_pred,
                    train_summary=train_summary,
                    valid_pred=valid_pred,
                    valid_summary=valid_summary)

                self._epoch_timer.split('lvl')

            else:
                valid_pred = valid_summary = None

            self._maybe_update_visualizations(
                train_data=epoch_train_data,
                train_pred=train_pred,
                train_summary=train_summary,
                valid_data=valid_data,
                valid_pred=valid_pred,
                valid_summary=valid_summary,
                version='seso')
            self._epoch_timer.split('visualize')

            done = self._is_training_complete(self.epoch_loss)
            self._epoch_timer.split('other', stop=True)

            self._print_epoch_summary(train_summary)

        self.timer.split('train')

        # There's an opportunity to save a call to predict in _close_training()
        # for the seso model--just pass it the train predictions from the last
        # epoch. Future work.
        epoch_train_data = self._prepare_epoch_data(train_data)
        self._close_training(epoch_train_data, valid_data)

        self.timer.split('_close_training')

        self._print_run_summary()

    def _setup_training(self, train_data, valid_data=None):
        '''Performs any tasks that must be completed before entering the
        training loop in self.train().

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        self._has_valid_data = valid_data is not None

    def _initialize_epoch(self):

        # Number of segments to time for profiling
        N_EPOCH_SPLITS = 8

        self._print_epoch_state()

        self._epoch_timer = Timer(N_EPOCH_SPLITS,
            name='Epoch',
            do_retrospective=True)
        self._epoch_timer.start()

    def _prepare_epoch_data(self, train_data):

        if train_data is None:
            # For on-the-fly data generation
            return self.generate_data('train')
        else:
            return train_data

    def _train_epoch(self, train_data):
        '''Performs training steps across an epoch of training data batches.

        Args:
            train_data: dict containing the training data. If not provided
            (i.e., train_data=None), the subclass implementation of
            _split_data_into_batches(...) must generate training data on the
            fly. Default: None.

        Returns:
            predictions: dict containing model predictions based on data. Key/
            value pairs will be specific to the subclass implementation.

            summary: dict containing high-level summaries of the predictions.
            Key/value pairs will be specific to the subclass implementation.
            Must contain key: 'loss' whose value is a scalar indicating the
            evaluation of the overall objective function being minimized
            during training.
        '''

        data_batches, batch_indices = self._split_data_into_batches(
            train_data)
        self._epoch_timer.split('data')

        pred_list = []
        summary_list = []

        for batch_data in data_batches:

            batch_pred, batch_summary = self._train_batch(batch_data)
            pred_list.append(batch_pred)
            summary_list.append(batch_summary)

        predictions, summary = self._combine_prediction_batches(
            pred_list, summary_list, batch_indices)

        self.prev_loss = self.epoch_loss
        self.epoch_loss = summary['loss']
        self.epoch_grad_norm = summary['grad_global_norm']

        ''' Note, these updates are intentionally placed before any
        possible checkpointing for the epoch. This placement is critical
        for reproducible training trajectories when restoring (i.e., for
        robustness to unexpected restarts). See note in
        _print_run_summary(). '''
        self._update_learning_rate()
        self._update_grad_clipping()
        self._increment_epoch()

        return predictions, summary

    def _update_learning_rate(self):

        self.adaptive_learning_rate.update(self.epoch_loss)

    def _update_grad_clipping(self):

        self.adaptive_grad_norm_clip.update(self.epoch_grad_norm)

    def _print_epoch_state(self, n_indent=1):
        ''' Prints information about the current epoch before any training
        steps have been taken within the epoch.

        Args:
            None.

        Returns:
            None.
        '''
        print('Epoch %d (step %d):' % (self._epoch, self._step))
        print('\t' * n_indent, end='')
        print('Learning rate: %.2e' % self.adaptive_learning_rate())

    def _print_epoch_summary(self, train_summary, n_indent=1):
        ''' Prints an summary describing one epoch of training.

        Args:
            train_summary: dict as returned by _train_batch()

        Returns:
            None.
        '''

        ''' At the time this is called, all training steps have been taken for
        the epoch, and those steps are reflected in the loss values (and other
        summary scalars) in batch_summaries.

        Additionally, the next set of updates have been applied to epoch,
        learning rate, and gradient clipping, but those updates have not
        yet influenced a gradient step on the model parameters. If desired,
        those values should be logged in _print_epoch_state(...), which is
        called before any training steps have been taken for the epoch.

        In other words, here we have a model/predictions/summaries from epoch
        n, but self._epoch(), self.adaptive_learning_rate(), and gradient
        clipping parameters have all been updated to their epoch n+1 values.

        This should not be changed, since it's critical for properly restoring
        a model and its training trajectory. '''

        if self.prev_loss is None:
            loss_improvement = np.nan
        else:
            loss_improvement = self.prev_loss - self.epoch_loss

        indent = '\t' * n_indent
        print('%sTraining loss: %.2e' % (indent, self.epoch_loss))
        print('%sImprovement: %.2e' % (indent, loss_improvement))
        print('%sLogging to: %s' % (indent, self.run_dir))
        self._epoch_timer.print(do_single_line=True, n_indent=n_indent)
        print('')

    def _print_run_summary(self):
        ''' Prints a final summary of the complete optimization.

        Args:
            None.

        Returns:
            None.
        '''

        print('')
        self.timer.print()
        print('')

    def _train_batch(self, batch_data):
        '''Runs one training step. This function must evaluate the Tensorboard
        summaries:
            self.tensorboard['merged_opt_summary']
            self.tensorboard['merged_hist_summary']

        Args:
            batch_data: dict containing one batch of training data. Key/value
            pairs will be specific to the subclass implementation.

        Returns:
            predictions: dict containing model predictions based on data. Key/
            value pairs will be specific to the subclass implementation.

            summary: dict containing summary data from this training
            step. Minimally, this includes the following key/val pairs:

                'loss': scalar float evaluation of the loss function over the
                data batch (i.e., an evaluation of self.loss).

                'grad_global_norm': scalar float evaluation of the norm of the
                gradient of the loss function with respect to all trainable
                variables, taken over the data batch (i.e., an evaluation of
                self.grad_global_norm).

        Note:
            Training predictions are never used by RecurrentWhisperer.
            Thus, if your subclass does not use them, it is safe to return them
            as None, which may reduce computation (although the predictions
            often come almost for free when running both the forwards and
            backwards passed through the model for gradient computations.

        '''
        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _is_training_complete(self, epoch_loss):
        '''Determines whether the training optimization procedure should
        terminate. Termination criteria, governed by hyperparameters, are
        thresholds on the following:

            1) the training loss
            2) the learning rate
            3) the number of training epochs performed
            4) the number of training epochs performed since the lowest
               validation loss improved (only if do_check_lvl == True).

        Args:
            epoch_summaries:

        Returns:
            bool indicating whether any of the termination criteria have been
            met.
        '''

        hps = self.hps

        complete = False

        if self.is_done(self.run_dir):
            print('Stopping optimization: found .done file.')
            complete = True

        elif epoch_loss is np.inf:
            print('\nStopping optimization: loss is Inf!')
            complete = True

        elif np.isnan(epoch_loss):
            print('\nStopping optimization: loss is NaN!')
            complete = True

        elif hps.min_loss is not None and epoch_loss <= hps.min_loss:
            print ('\nStopping optimization: loss meets convergence criteria.')
            complete = True

        elif self.adaptive_learning_rate.is_finished(do_check_step=False):
            print ('\nStopping optimization: minimum learning rate reached.')
            complete = True

        elif self.adaptive_learning_rate.is_finished(do_check_rate=False):
            print('\nStopping optimization:'
                  ' reached maximum number of training epochs.')
            complete = True

        elif hps.max_train_time is not None and \
            self._train_time > hps.max_train_time:

            print ('\nStopping optimization: training time exceeds '
                'maximum allowed.')
            complete = True

        elif self._has_valid_data:
            # Check whether lvl has been given a value (after being
            # initialized to np.inf), and if so, check whether that value has
            # improved recently.
            if not np.isinf(self._lvl) and \
                self._epoch - self._epoch_last_lvl_improvement >= \
                    hps.max_n_epochs_without_lvl_improvement:

                print('\nStopping optimization:'
                      ' reached maximum number of training epochs'
                      ' without improvement to the lowest validation loss.')

                complete = True

        return complete

    def _close_training(self, train_data=None, valid_data=None):
        ''' Optionally saves a final checkpoint, then loads the LVL model and
        generates LVL visualizations.

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''
        hps = self.hps

        print('\nClosing training:')

        # Save checkpoint upon completing training
        if hps.do_save_seso_ckpt:
            self._save_checkpoint(version='seso')

        # Save .done file. Critically placed after saving final checkpoint,
        # but before doing a bunch of other stuff that might fail. This way,
        # if anything below does fail, the .done file will be present,
        # indicating safe to interpret checkpoint model as final.
        # Also, subclass implementations of predict(), update_visualizations(),
        # etc can check self.is_done to do certain expensive things just once
        # at the end of training, rather than on every call throughout
        # training.
        self._save_done_file()

        self.save_final_results(train_data, valid_data, version='seso')
        self.save_final_results(train_data, valid_data, version='lvl')
        self.save_final_results(train_data, valid_data, version='ltl')

        if hps.do_log_output:
            self._log_file.close()

            # Redirect all printing, errors, and warnings back to defaults
            sys.stdout = self._default_stdout
            sys.stderr = self._default_stderr

    def save_final_results(self, train_data, valid_data, version='seso'):
        ''' Optionally save predictions and/or visualizations upon completion
        of training. This will optionally restore from 'ltl' or 'lvl'
        checkpoints, or will use the current model state if version is 'seso'.
        '''

        def _do_train_visualizations(version, data):

            if data is None:
                return False

            if version == 'seso':
                return self.hps.do_generate_final_visualizations
            elif version == 'lvl':
                return False
            elif version == 'ltl':
                return self.hps.do_generate_ltl_visualizations

        def _do_valid_visualizations(version, data):

            if data is None:
                return False

            if version == 'seso':
                return self.hps.do_generate_final_visualizations
            elif version == 'lvl':
                return self.hps.do_generate_lvl_visualizations
            elif version == 'ltl':
                return False

        def _do_train_predictions(version, data):

            if data is None:
                return False

            if version == 'seso':
                return False

            return self._do_save_pred('train', version)

        def _do_valid_predictions(version, data):

            if data is None:
                return False

            if version == 'seso':
                return False

            return self._do_save_pred('valid', version)

        def _do_save_visualizations(version):

            hps = self.hps
            if version == 'seso':
                return hps.do_save_final_visualizations
            elif version == 'ltl':
                return hps.do_save_ltl_visualizations
            elif version == 'lvl':
                return hps.do_save_lvl_visualizations

        self._assert_ckpt_version(version)

        do_train_pred = _do_valid_predictions(version, train_data)
        do_train_vis = _do_train_visualizations(version, train_data)

        do_valid_pred = _do_valid_predictions(version, valid_data)
        do_valid_vis = _do_valid_visualizations(version, valid_data)

        do_list = [do_train_pred, do_train_vis, do_valid_pred, do_valid_vis]

        if not any(do_list):
            # This covers the case of data generated on the fly
            # (train_data=None, valid_data=None)
            return

        if version in ['ltl' or 'lvl']:
            self.restore_from_checkpoint(version)

        if do_train_pred or do_train_vis:

            train_pred, train_summary = self.predict(train_data,
                train_or_valid_str='train')

            if do_train_pred:
                self._save_pred(train_pred, 'train', version=version)
                self._save_summary(train_summary, 'train', version=version)

            if do_train_vis:
                self.update_visualizations(
                    train_data, train_pred, 'train', version,
                    do_save=_do_save_visualizations(version))

        if do_valid_pred or do_valid_vis:

            valid_pred, valid_summary = self.predict(valid_data,
                train_or_valid_str='valid')

            if do_valid_pred:
                self._save_pred(valid_pred, 'valid', version=version)
                self._save_summary(valid_summary, 'valid', version=version)

            if do_valid_vis:
                self.update_visualizations(
                    valid_data, valid_pred, 'valid', version,
                    do_save=_do_save_visualizations(version))

    _batch_size_key = 'batch_size'

    # *************************************************************************
    # Prediction **************************************************************
    # *************************************************************************

    def predict(self, data,
        train_or_valid_str='valid',
        do_batch=None,
        is_final=False):
        ''' Runs a forward pass through the model using given input data. If
        the input data are larger than the batch size, the data are processed
        sequentially in multiple batches.

        Args:
            data: dict containing requisite data for generating predictions.
            Key/value pairs will be specific to the subclass implementation.

            train_or_valid_str (optional): either 'train' or 'valid',
            indicating whether data contains training data or validation data,
            respectively. This can be used to modulate the behavior of
            _predict_batch(). Default: 'valid'.

            do_batch (optional): bool indicating whether to split data into
            batches and then sequentially process those batches. This can be
            important for large models and/or large datasets relative to
            memory resources. Default: falls back to hps.do_batch_predictions.

            is_final (optional): bool indicating

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

        if do_batch is None:
            do_batch = self.hps.do_batch_predictions

        if do_batch:

            batches_list, batch_indices = self._split_data_into_batches(data)
            n_batches = len(batches_list)
            pred_list = []
            summary_list = []
            for cnt, batch_data in enumerate(batches_list):

                batch_size = self._get_batch_size(batch_data)

                print('\t\tPredict: batch %d of %d (%d trials)'
                      % (cnt+1, n_batches, batch_size))

                batch_pred, batch_summary = self._predict_batch(
                    batch_data, train_or_valid_str=train_or_valid_str)

                pred_list.append(batch_pred)
                summary_list.append(batch_summary)

            predictions, summary = self._combine_prediction_batches(
                pred_list, summary_list, batch_indices)

        else:
            predictions, summary = self._predict_batch(
                data, train_or_valid_str=train_or_valid_str)

        assert ('loss' in summary),\
            ('summary must minimally contain key: \'loss\', but does not.')

        return predictions, summary

    def _predict_batch(self, batch_data, train_or_valid_str='valid'):
        ''' Runs a forward pass through the model using a single batch of data.

        Args:
            batch_data: dict containing requisite data for generating
            predictions. Key/value pairs will be specific to the subclass
            implementation.

            train_or_valid_str (optional): either 'train' or 'valid',
            indicating whether data contains training data or validation data,
            respectively. Default: 'valid'.

        Returns:
            predictions: See docstring for predict().

            summary:  See docstring for predict().
        '''

        raise StandardError(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    # *************************************************************************
    # Validation **************************************************************
    # *************************************************************************

    @property
    def _do_predict_validation(self):
        ''' Returns true if validation predictions or prediction summary
        is needed at the current epoch. '''

        return self._has_valid_data and \
            (self._do_update_validation or self._do_update_visualizations)

    @property
    def _do_update_validation(self):
        n = self.hps.n_epochs_per_lvl_update
        return np.mod(self._epoch, n) == 0

    # *************************************************************************
    # Data and batch management ***********************************************
    # *************************************************************************

    def generate_data(self, train_or_valid_str='train'):
        ''' Optionally generate data on-the-fly (e.g., during training), rather
        than relying on fixed sets of training and validation data. This is only called by train(...) when called using train_data=None.

        Args:
            train_or_valid_str:

        Returns:
            data: dict containing the generated data.
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

    def _split_data_into_batches(self, data):
        ''' Randomly splits data into a set of batches. If the number of
        trials in data evenly divides by max_batch_size, all batches have size
        max_batch_size. Otherwise, the last batch is smaller containing the
        remainder trials.

        Args:
            data: dict containing the to-be-split data.

        Returns:
            data_batches: list of dicts, where each dict contains one batch of
            data.

            batch_indices: list, where each element, idx_list[i], is a list of
            the trial indices for the corresponding batch of data in data_list[
            i]. This is used to recombine the trials back into their original
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

            batch_idx: array-like of trial indices.

        Returns:
            subselected_data: dict containing the subselected data.
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

    @classmethod
    def _combine_batch_summaries(cls, batch_summaries):
        ''' Combines batched results from _train_batch(...) into a single
        summary dict, formatted identically to an individual batch_summary.

        For each summary scalar, this is done by averaging that scalar across
        all batches, weighting by batch size. The only exception is batch_size
        itself, which is summed.

        NOTE: A combined value will only be interpretable if that value in an
        individual original batch_summary is itself an average across that
        batch (as opposed to it being a sum across that batch).

        Args:
            batch_summaries:
                List of summary dicts, as returned by _train_epoch(...).

        Returns:
            summary:
                A single summary dict with the same keys as those in each of
                the batch_summaries.
        '''

        BATCH_SIZE_KEY = cls._batch_size_key
        summary = {}

        # Average everything except batch_size
        for key in np.sort(batch_summaries[0].keys()):

            if key == BATCH_SIZE_KEY:
                pass
            else:
                summary[key] = cls._compute_epoch_average(batch_summaries, key)

        # Sum batch sizes
        batch_size = np.sum([s[BATCH_SIZE_KEY] for s in batch_summaries])
        summary[BATCH_SIZE_KEY] = batch_size

        return summary

    @classmethod
    def _compute_epoch_average(cls, batch_summaries, key):
        '''Computes a weighted average of evaluations of a summary
        statistic across an epoch of data batches. This is all done in
        numpy (no Tensorflow).

        Args:
            batch_summaries: list of dicts, with each dict as returned by
            _train_batch() and updated to include 'batch_size' (done
            automatically in _train_epoch).

            key: string name of the statistic in each batch summary dict, whose
            values are to be averaged.

        Returns:
            avg: float or numpy array containing the batch-size-weighted
            average of the batch_summaries[i][key] values. Shape matches that
            of each batch_summaries[i][key] value (typically a scalar).
        '''

        BATCH_SIZE_KEY = cls._batch_size_key
        batch_vals = []
        batch_sizes = []

        assert isinstance(batch_summaries, list),\
            ('batch_summaries must be a list, '
             'but has type: %s' % str(type(batch_summaries)))

        assert len(batch_summaries)>=0,\
            'Cannot compute epoch averages because batch_summaries is empty.'

        for batch_summary in batch_summaries:

            assert key in batch_summary,\
                ('Did not find key (%s) in batch_summary.' % key)

            assert BATCH_SIZE_KEY in batch_summary,\
                ('Did not find key (%s) in batch_summary.' % BATCH_SIZE_KEY)

            batch_vals.append(batch_summary[key])
            batch_sizes.append(batch_summary[BATCH_SIZE_KEY])

        # Deprecated. Only works if batch_summary[key] is scalar.
        # weights = np.true_divide(batch_sizes, np.sum(batch_sizes))
        # avg = np.dot(weights, batch_vals)

        # This supports arbitrary shaped numpy arrays
        # (though by convention only scalars should be in prediction summary.)
        avg = np.average(batch_vals, weights=batch_sizes, axis=0)

        return avg

    # *************************************************************************
    # Visualizations **********************************************************
    # *************************************************************************

    def _update_visualizations(self, data, pred, train_or_valid_str, version):
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
        pass

    def save_visualizations(self, version='seso'):
        '''Saves individual figures to this run's figure directory. This is
        independent of Tensorboard Images.

        Args:
            None.

        Returns:
            None.
        '''
        hps = self.hps
        fig_dir = self._build_fig_dir(self.run_dir, version=version)

        for fig_name, fig in self.figs.iteritems():

            self._visualizations_timer.split('Saving: %s' % fig_name)

            file_path_no_ext = os.path.join(fig_dir, fig_name)
            figs_dir_i, file_name_no_ext = os.path.split(file_path_no_ext)
            file_name = file_name_no_ext + '.' + hps.fig_filetype
            file_path = os.path.join(figs_dir_i, file_name)

            # This fig's dir may have additional directory structure beyond
            # the already existing .../figs/ directory. Make it.
            if not os.path.isdir(figs_dir_i):
                os.makedirs(figs_dir_i)

            fig.savefig(file_path,
                bbox_inches='tight',
                format=hps.fig_filetype,
                dpi=hps.fig_dpi)

        # Make sure whatever happens next doesn't affect timing of last save.
        self._visualizations_timer.split('Transition from saving.')

    def _maybe_update_visualizations_pretraining(self,
        train_data, valid_data):

        # Visualizations generated from untrained network
        if self._do_update_pretraining_visualizations:

            epoch_train_data = self._prepare_epoch_data(train_data)
            train_pred, train_summary = self.predict(
                epoch_train_data, train_or_valid_str='train')

            if valid_data:
                valid_pred, valid_summary = self.predict(
                    valid_data, train_or_valid_str='valid')
            else:
                valid_pred = valid_summary = None

            self.update_visualizations(
                train_data=epoch_train_data,
                train_pred=train_pred,
                train_summary=train_summary,
                valid_data=valid_data,
                valid_pred=valid_pred,
                valid_summary=valid_summary,
                do_save=self.hps.do_save_pretraining_visualizations)

    def _maybe_update_visualizations(self, **kwargs):
        '''Updates visualizations if the current epoch number indicates that
        an update is due. Saves those visualization to Tensorboard or to
        individual figure files, depending on hyperparameters
        (do_save_tensorboard_images and do_save_training_visualizations,
        respectively.)

        Args:
            train_data:
            train_pred:
            train_summary:
            valid_data:
            valid_pred:
            valid_summary:
            version:

        Returns:
            None.
        '''

        if self._do_update_pretraining_visualizations:
            do_save = self.hps.do_save_pretraining_visualizations
        elif self._do_update_training_visualizations:
            do_save = self.hps.do_save_training_visualizations
        else:
            return

        self.update_visualizations(do_save=do_save, **kwargs)

    def update_visualizations(self,
        train_data=None,
        train_pred=None,
        train_summary=None,
        valid_data=None,
        valid_pred=None,
        valid_summary=None,
        version='seso',
        do_save=True, # Save individual figures (indep of Tensorboard)
        do_update_tensorboard=None, # default: hps.do_save_tensorboard_images
        ):

        self._setup_visualizations_timer()

        if train_data and train_pred:
            self._update_visualizations(
                data=train_data,
                pred=train_pred,
                train_or_valid_str='train',
                version=version)

        if valid_data and valid_pred:
            self._update_visualizations(
                data=valid_data,
                pred=valid_pred,
                train_or_valid_str='valid',
                version=version)

        if do_update_tensorboard is None:
            do_update_tensorboard = self.hps.do_save_tensorboard_images

        if do_update_tensorboard:
            self._update_tensorboard_images()

        if do_save:
            self.save_visualizations(version=version)

        self._maybe_print_visualizations_timing()

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

        self._visualizations_timer.split('Plotting: %s' % fig_name)

        if fig_name not in self.figs:
            self.figs[fig_name] = plt.figure(
                figsize=(width, height),
                tight_layout=tight_layout)

        fig = self.figs[fig_name]
        fig.clf()

        return fig

    def _setup_visualizations_timer(self):

        # Future work: this could be made more conservative.
        n_splits = max(100, 4 + 2 * self.n_figs)

        self._visualizations_timer = Timer(n_splits,
            name='Visualizations',
            do_retrospective=False,
            n_indent=2)

        self._visualizations_timer.split('Initial prep')

    def _maybe_print_visualizations_timing(self):

        if self.hps.do_print_visualizations_timing:
            self._visualizations_timer.print()

    @property
    def _do_update_visualizations(self):
        return self._do_update_pretraining_visualizations or \
            self._do_update_training_visualizations

    @property
    def _do_update_pretraining_visualizations(self):

        hps = self.hps

        if not hps.do_generate_pretraining_visualizations:
            # Avoid getting epoch from TF graph
            return False

        return self._epoch == 0

    @property
    def _do_update_training_visualizations(self):

        hps = self.hps

        if not hps.do_generate_training_visualizations:
            # Avoid getting epoch from TF graph
            return False

        epoch = self._epoch

        return  epoch > 0 and \
            np.mod(epoch, hps.n_epochs_per_visualization_update) == 0

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

    @property
    def n_figs(self):
        return len(self.figs)

    # *************************************************************************
    # Exposed directory access ************************************************
    # *************************************************************************

    @property
    def run_hash(self):
        return self._run_hash

    @property
    def run_dir(self):
        return self._run_dir

    @property
    def events_dir(self):
        return self._subdirs['events']

    @property
    def seso_dir(self):
        return self._subdirs['seso']

    @property
    def ltl_dir(self):
        return self._subdirs['ltl']

    @property
    def lvl_dir(self):
        return self._subdirs['lvl']

    @property
    def hps_dir(self):
        return self._subdirs['hps']

    @property
    def fps_dir(self):
        return self._subdirs['fps']

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
            list of cross-validation runs (folder names) found in run_dir.

        '''

        def list_dirs(path_str):
            return [name for name in os.listdir(path_str) \
                if os.path.isdir(os.path.join(path_str, name)) ]

        run_info = []
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

    @classmethod
    def is_run_dir(cls, run_dir):
        '''Determines whether a run exists in a specified directory.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__().

        Returns:
            bool indicating whether a run exists.
        '''

        if run_dir is None:
            return False

        # Check for existence of all directories that would have been created
        # if a run was executed. This won't look for various files, which may
        # or may not be saved depending on hyperparameter choices.
        dirs = cls._build_subdirs(run_dir)
        exists = [os.path.exists(d) for d in dirs.values()]
        return all(exists)

    @classmethod
    def is_done(cls, run_dir):
        '''Determines whether a run exists in the filesystem and has run to
        completion.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__().

        Returns:
            bool indicating whether the run is "done".
        '''

        done_path = cls._build_done_path(run_dir)
        return os.path.exists(done_path)

    # *************************************************************************
    # Scalar access and updates ***********************************************
    # *************************************************************************
    #
    # Convention: properties beginning with an underscore return numpy or
    # python numeric types (e.g., _epoch, _lvl). The corresponding TF
    # Variables are in self.records. See _setup_records().

    @property
    def trainable_variables(self):
        ''' Returns the list of trainable TF variables that compose this model.

        Args:
            None.

        Returns:
            A list of TF.Variable objects.
        '''

        # Exclude anything a user may have created on the graph that is not
        # part of this model.
        tf_vars = tf.trainable_variables()
        model_vars = [v for v in tf_vars if self.hps.name in v.name]

        return model_vars

    @property
    def _ltl(self):
        '''Returns the lowest training loss encountered thus far (i.e., across
        an entire pass through the training data).

        Args:
            None.

        Returns:
            ltl: float specifying the lowest training loss.
        '''

        # TO DO: remove "_" from definition
        return self.session.run(self.records['ops']['ltl'])

    @property
    def _lvl(self):
        '''Returns the lowest validation loss encountered thus far during
        training.

        Args:
            None.

        Returns:
            lvl: float specifying the lowest validation loss.
        '''

        # TO DO: remove "_" from definition
        return self.session.run(self.records['ops']['lvl'])

    @property
    def _train_time(self):
        '''Returns the time elapsed during training, measured in seconds, and
        accounting for restoring from previously saved runs.

        Args:
            None.

        Returns:
            float indicating time elapsed during training..
        '''

        # TO DO: remove "_" from definition
        return self.train_time_offset + self.timer()

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

        # TO DO: remove "_" from definition
        return self.session.run(self.records['ops']['global_step'])

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

        # TO DO: remove "_" from definition
        return self.session.run(self.records['ops']['epoch'])

    @property
    def _epoch_next_ltl_check(self):
        hps = self.hps
        return self._epoch_last_ltl_improvement + hps.n_epochs_per_ltl_update

    @property
    def _epoch_last_ltl_improvement(self):
        '''Returns the epoch of the most recent improvement to the lowest loss
        over the training data.

        Args:
            None.

        Returns:
            int specifying the epoch number.
        '''
        return self._epoch_last_loss_improvement('ltl')

    @property
    def _epoch_last_lvl_improvement(self):
        '''Returns the epoch of the most recent improvement to the lowest loss
        over the validation data.

        Args:
            None.

        Returns:
            int specifying the epoch number.
        '''
        return self._epoch_last_loss_improvement('lvl')

    def _epoch_last_loss_improvement(self, version):

        self._assert_version_is_ltl_or_lvl(version)

        op_name = 'epoch_last_%s_improvement' % version
        return self.session.run(self.records['ops'][op_name])

    def _update_train_time(self):
        '''Runs the TF op that updates the time elapsed during training.

        Args:
            None.

        Returns:
            None.
        '''
        time_val = self._train_time
        self.session.run(
            self.records['update_ops']['train_time'],
            feed_dict={self.records['placeholders']['train_time']: time_val})

    def _increment_epoch(self):
        self.session.run(self.records['increment_ops']['epoch'])

    def _update_loss_records(self, loss, version, epoch=None):
        ''' Updates TF records of the lowest loss and the epoch in which this
        improvement was achieved. This is critical for maintaining the
        trajectory of training across checkpoint saves and restores--i.e., for
        robustness to restarts.

        Args:
            loss: A numpy scalar value indicating the (new) lowest loss.

            version: 'ltl' or 'lvl', indicating whether this is the lowest
            training loss or lowest validation loss, respectively.

            epoch (optional): Numpy scalar indicating the epoch of this
            improvement. Default: the current epoch.

        Returns:
            None.
        '''

        self._assert_version_is_ltl_or_lvl(version)

        # Not Tensorflow
        if epoch is None:
            epoch = self._epoch

        # E.g., 'epoch_last_lvl_improvement'
        epoch_key = 'epoch_last_%s_improvement' % version

        # Tensorflow
        placeholders = self.records['placeholders']
        update_ops = self.records['update_ops']

        feed_dict = {
            placeholders[version]: loss,
            placeholders[epoch_key]: epoch}
        ops = [
            update_ops[version],
            update_ops[epoch_key]
            ]

        self.session.run(ops, feed_dict=feed_dict)

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
        model_vars = self.trainable_variables
        n_params = sum([np.prod(v.shape).value for v in model_vars])

        return n_params

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
            self._update_loss_records(np.inf, version='ltl')
            self._update_loss_records(np.inf, version='lvl')

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
            zipped_grads, global_step=self.records['ops']['global_step'])

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

    # *************************************************************************
    # Loading hyperparameters *************************************************
    # *************************************************************************

    @classmethod
    def load_hyperparameters(cls, run_dir):
        '''Load previously saved Hyperparameters.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing the loaded hyperparameters.
        '''

        hps_path = cls._build_hps_path(run_dir)

        if os.path.exists(hps_path):
            hps_dict = Hyperparameters.restore(hps_path)
        else:
            raise IOError('%s not found.' % hps_path)

        return hps_dict

    @classmethod
    def exists_hyperparameters(cls, run_dir):
        hps_path = cls._build_hps_path(run_dir)
        return os.path.exists(hps_path)

    # *************************************************************************
    # Saving: model checkpoints ***********************************************
    # *************************************************************************

    def _maybe_save_seso_checkpoint(self):
        '''Saves a model checkpoint if the current epoch number indicates that
        a checkpoint is due.

        Args:
            None.

        Returns:
            None.
        '''
        if self._do_save_seso_checkpoint:
            self._save_checkpoint(version='seso')

    def _maybe_save_ltl_checkpoint(self, train_pred, train_summary):
        ''' Saves a model checkpoint if the current training loss is lower than
        all previously evaluated training losses.

        Args:
            train_pred and train_summary: dicts as returned by
            predict(train_data).

        Returns:
            None.
        '''

        version = 'ltl'
        train_loss = self.epoch_loss

        if self._do_save_ltl_checkpoint(train_loss):

            self._update_loss_records(train_loss, version=version)

            if self.hps.do_save_ltl_ckpt:
                self._save_checkpoint(version=version)

            if self.hps.do_save_ltl_train_summary:
                self._save_summary(train_summary, 'train', version=version)

    def _maybe_save_lvl_checkpoint(self,
        train_pred,
        train_summary,
        valid_pred,
        valid_summary):
        '''Saves a model checkpoint if the current validation loss is lower
        than all previously evaluated validation losses. Optionally, this will
        also generate and save model predictions over the training and
        validation data.

        Args:
            train_pred and train_summary: dicts as returned by
            predict(train_data).

            valid_pred and valid_summary: dicts as returned by
            predict(valid_data).

        Returns:
            None.
        '''

        version = 'lvl'
        valid_loss = valid_summary['loss']

        if self._do_save_lvl_checkpoint(valid_loss):

            print('\tValidation loss: %.2e' % valid_loss)
            self._update_loss_records(valid_loss, version=version)

            if self.hps.do_save_lvl_ckpt:
                self._save_checkpoint(version=version)

            self._maybe_save_pred_and_summary('train',
                pred=train_pred,
                summary=train_summary,
                version=version)

            self._maybe_save_pred_and_summary('valid',
                pred=valid_pred,
                summary=valid_summary,
                version=version)

        if self.hps.do_save_tensorboard_summaries:
            self._update_valid_tensorboard_summaries(valid_summary)

    # *************************************************************************
    # Forward thinking, not yet using these ***********************************
    # *************************************************************************
    def _maybe_save_checkpoint(self, data, pred, summary, version):
        '''Saves a model checkpoint if the current validation loss is lower
        than all previously evaluated validation losses. Optionally, this will
        also generate and save model predictions over the training and
        validation data.

        If prediction summaries are generated, those summaries are saved in
        separate files. See docstring for predict() for additional detail.

        Args:
            valid_pred and valid_summary: dicts as returned by
            predict(valid_data).

            train_data: dict containing the training data.

        Returns:
            None.
        '''

        loss = summary['loss']

        if self._do_save_checkpoint(loss, version):

            self._update_loss_records(valid_loss, version=version)

            if self.hps.do_save_lvl_ckpt:
                self._save_checkpoint(version=version)

            self._maybe_save_pred_and_summary('train',
                data=train_data,
                version=version)

            self._maybe_save_pred_and_summary('valid',
                pred=valid_pred,
                summary=valid_summary,
                version=version)

        self._epoch_timer.split(version)

    ''' Currently there's a bit of asymmetry: ltl and lvl check
    hps.do_save_*_ckpt upstream, but hps.do_save_seso_ckpt is checked here. '''
    @property
    def _do_save_seso_checkpoint(self):

        n = self._epoch
        n_per_update = self.hps.n_epochs_per_seso_update

        return self.hps.do_save_seso_ckpt and np.mod(n, n_per_update) == 0

    def _do_save_ltl_checkpoint(self, train_loss):

        epoch = self._epoch

        if epoch == 0 or \
            (train_loss < self._ltl and epoch >= self._epoch_next_ltl_check):

            print('\tAchieved lowest training loss.')
            return True

        else:
            return False

    def _do_save_lvl_checkpoint(self, valid_loss):

        if self._epoch == 0 or valid_loss < self._lvl:
            print('\t\tAchieved lowest validation loss.')
            return True
        else:
            return False

    # *************************************************************************
    # *************************************************************************
    # *************************************************************************

    def _save_checkpoint(self, version):
        '''Saves a model checkpoint, along with data for restoring the adaptive
        learning rate and the adaptive gradient clipper.

        Args:
            version: string indicating which version to label this checkpoint
            as: 'seso', 'ltl', or 'lvl'.

        Returns:
            None.
        '''

        self._assert_ckpt_version(version)

        print('\t\tSaving %s checkpoint.' % str.upper(version))
        ckpt_path = self._get_ckpt_path_stem(version)

        self._update_train_time()
        saver = self.savers[version]
        saver.save(self.session, ckpt_path,
            global_step=self.records['ops']['global_step'])

        ckpt_dir, ckpt_fname = os.path.split(ckpt_path)
        self.adaptive_learning_rate.save(ckpt_dir)
        self.adaptive_grad_norm_clip.save(ckpt_dir)

    def _get_ckpt_dir(self, version):

        # E.g., self._subdirs['lvl']'
        return self._subdirs[version]

    def _get_ckpt_path_stem(self, version):

        # E.g., self._paths['lvl_ckpt_path']
        # Actual checkpoint path will append step and extension
        # (for that, use _get_ckpt_path, as relevant for restoring from ckpt)
        return self._paths['%s_ckpt_path' % version]

    @staticmethod
    def _assert_ckpt_version(version):
        assert version in ['seso', 'ltl', 'lvl'], \
            'Unsupported version: %s' % str(version)

    @staticmethod
    def _assert_version_is_ltl_or_lvl(version):
        assert version in ['ltl', 'lvl'], \
            'Unsupported version: %s' % str(version)

    @staticmethod
    def _assert_filetype(filetype):
        assert filetype in ['h5', 'npz', 'json', 'mat', 'pkl'],\
            'Unsupported filetype from extension: %s' % extension

    # *************************************************************************
    # Restoring from model checkpoints ****************************************
    # *************************************************************************

    @classmethod
    def restore(cls, run_dir, version, do_update_base_path=False):
        ''' Load a saved model given only the run directory, properly handling
        subclassing.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__().

            version: 'ltl', 'lvl', or 'seso' indicating which version of the
            model to load: lowest-training-loss, lowest-validation-loss, or
            'save-every-so-often', respectively. Which of these models exist,
            if any, depends on the hyperparameter settings used during
            training.

            do_update_base_path (optional): bool indicating whether to update
            all relevant filesystem paths using the directory structure
            inferred from run_dir. Set to True when restoring a model from a
            location other than where it was originally created/fit, e.g., if
            it points to a remote directory that is mounted locally or if the
            run directory was copied from another location. Default: False.

        Returns:
            The desired model with restored parameters, including the training
            state (epoch number, training time, adaptive learning rate,
            adaptive gradient clipping).
        '''

        # Validate version here. Existence of checkpoint is validated in
        # restore_from_checkpoint(...).
        cls._assert_ckpt_version(version)

        hps_dict = cls.load_hyperparameters(run_dir)
        log_dir = hps_dict['log_dir']

        if not do_update_base_path:
            # Assume standard checkpoint directory structure
            ckpt_path = None
        else:
            # Handle loading a model that was created/fit somewhere else, but
            # now is accessible via run_dir.

            # Get rid of trailing sep
            if run_dir[-1] == '/':
                run_dir = run_dir[:-1]

            # These are now relative to run_dir (which is local)
            log_dir, run_hash = os.path.split(run_dir)

            ckpt_dir = cls._build_subdir(run_dir, version)
            ckpt_path = cls._get_ckpt_path(ckpt_dir, do_update_base_path=True)

        # Build model but don't initialize any parameters, and don't restore
        # from standard checkpoints.
        hps_dict['log_dir'] = log_dir
        hps_dict['do_custom_restore'] = True
        hps_dict['do_log_output'] = False
        model = cls(**hps_dict)

        # Find and resotre parameters from lvl checkpoint
        model.restore_from_checkpoint(version, checkpoint_path=ckpt_path)

        return model

    def exists_checkpoint(self, version):
        '''
        Args:
            version: string indicating which version to label this checkpoint
            as: 'seso', 'ltl', or 'lvl'.
        '''

        self._assert_ckpt_version(version)
        ckpt_dir = self._get_ckpt_dir(version)

        # This might be equivalent.
        # ckpt_path = self._get_ckpt_path(version)
        # return tf.train.checkpoint_exists(ckpt_path)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        return ckpt is not None

    def restore_from_checkpoint(self, version,
        checkpoint_path=None):
        ''' Restores a model and relevant support structures from the most
        advanced previously saved checkpoint. This includes restoring TF model
        parameters, as well as adaptive learning rate (and history) and
        adaptive gradient clipping (and history).

        Args:
            version: 'ltl', 'lvl', or 'seso' indicating which version of the
            model to load: lowest-training-loss, lowest-validation-loss, or
            'save-every-so-often', respectively. Which of these models exist,
            if any, depends on the hyperparameter settings used during
            training.

            checkpoint_path (optional): string containing a path to
            a model checkpoint. Use this as an override if needed for
            loading models that were saved under a different directory
            structure (e.g., on another machine). Default: None.

        Returns:
            None.

        Raises:
            AssertionError if no checkpoint exists.
        '''

        self._assert_ckpt_version(version)

        if checkpoint_path is None:
            # Find ckpt path and recurse
            ckpt_dir = self._get_ckpt_dir(version)
            ckpt_path = self._get_ckpt_path(ckpt_dir)
            return self.restore_from_checkpoint(version,
                checkpoint_path=ckpt_path)
        else:
            assert tf.train.checkpoint_exists(checkpoint_path),\
                ('Checkpoint does not exist: %s' % checkpoint_path)

            ckpt_dir, ckpt_filename = os.path.split(checkpoint_path)

            # This is what we came here for.
            print('Loading checkpoint: %s.' % ckpt_filename)

            saver = self.savers[version]
            saver.restore(self.session, checkpoint_path)
            self.adaptive_learning_rate.restore(ckpt_dir)
            self.adaptive_grad_norm_clip.restore(ckpt_dir)

            # Resume training timer from value at last save.
            self.train_time_offset = self.session.run(
                self.records['ops']['train_time'])

    @classmethod
    def _get_ckpt_path(cls, ckpt_dir, do_update_base_path=False):

        ckpt = tf.train.get_checkpoint_state(ckpt_dir) # None if no ckpt
        assert ckpt is not None, ('No checkpoint found in: %s' % ckpt_dir)
        ckpt_path = ckpt.model_checkpoint_path

        if do_update_base_path:
            ''' If model was originally created/fit on a different machine, TF
            will refer to a bunch of paths that we no longer want to use. We
            only want to use the directory structure indicated in ckpt_dir,
            which is always local (because we made it through the assert
            above).
            '''
            prev_ckpt_dir, ckpt_filename = os.path.split(ckpt_path)
            ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

        return ckpt_path

    # *************************************************************************
    # Saving and loading: predictions and summaries ***************************
    # *************************************************************************

    @classmethod
    def exists_train_predictions(cls, run_dir, version='lvl'):
        return cls._exists_file(run_dir,
            train_or_valid_str='train',
            predictions_or_summary_str='predictions',
            version=version)

    @classmethod
    def exists_train_summary(cls, run_dir, version='lvl'):
        return cls._exists_file(run_dir,
            train_or_valid_str='train',
            predictions_or_summary_str='summary',
            version=version)

    @classmethod
    def exists_valid_predictions(cls, run_dir, version='lvl'):
        return cls._exists_file(run_dir,
            train_or_valid_str='valid',
            predictions_or_summary_str='predictions',
            version=version)

    @classmethod
    def exists_valid_summary(cls, run_dir, version='lvl'):
        return cls._exists_file(run_dir,
            train_or_valid_str='valid',
            predictions_or_summary_str='summary',
            version=version)

    def save_predictions_and_summary(self, data, train_or_valid_str, version):
        ''' Saves model predictions and a prediction summary, regardless of the
        hyperparameters. This is provided for external convenience, and is
        never used internally.

        Prediction summaries are saved in a separate .pkl file from the
        predictions themselves. See docstring for predict() for additional
        detail.

        Args:
            data: dict containing the data over which predictions are
            generated. This can be the training data or the validation data.

            train_or_valid_str: either 'train' or 'valid', indicating whether
            data contains training data or validation data, respectively.

            version: 'ltl', 'lvl', or 'seso' indicating whether the state of
            the model is lowest-training-loss, lowest-validation-loss, or
            'save-every-so-often', respectively. This determines the names and
            locations of the files to be saved.

        Returns:
            None.
        '''

        self._assert_ckpt_version(version)

        pred, summary = self.predict(data,
            train_or_valid_str=train_or_valid_str)
        self._save_pred(pred, train_or_valid_str, version=version)
        self._save_summary(summary, train_or_valid_str, version=version)

    @classmethod
    def load_train_predictions(cls, run_dir,
        version='lvl',
        filetype='npz'):
        '''Loads predictions made over the training data by a specified
        checkpoint of the model.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing saved predictions.
        '''
        return cls._load_pred_or_summary_helper(run_dir,
            train_or_valid_str='train',
            predictions_or_summary_str='predictions',
            version=version,
            filetype=filetype)

    @classmethod
    def load_train_summary(cls, run_dir,
        version='lvl',
        filetype='npz'):
        '''Loads summary of the model predictions made over the training
         data by a specified checkpoint of the model.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing saved summaries.
        '''
        return cls._load_pred_or_summary_helper(run_dir,
            train_or_valid_str='train',
            predictions_or_summary_str='summary',
            version=version,
            filetype=filetype)

    @classmethod
    def load_valid_predictions(cls, run_dir,
        version='lvl',
        filetype='npz'):
        '''Loads predictions made over the validation data by a specified
        checkpoint of the model.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing saved predictions.
        '''
        return cls._load_pred_or_summary_helper(run_dir,
            train_or_valid_str='valid',
            predictions_or_summary_str='predictions',
            version=version,
            filetype=filetype)

    @classmethod
    def load_valid_summary(cls, run_dir,
        version='lvl',
        filetype='npz'):
        '''Loads summary of the model predictions made over the validation
         data by a specified checkpoint of the model.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing saved summaries.
        '''
        return cls._load_pred_or_summary_helper(run_dir,
            train_or_valid_str='valid',
            predictions_or_summary_str='summary',
            version=version,
            filetype=filetype)

    def _maybe_save_pred_and_summary(self, train_or_valid_str,
        data=None,    # Either provide data, ...
        pred=None,    # or provide both pred
        summary=None, # and summary.
        version='lvl'):
        '''Saves model predictions and/or a prediction summary. Which are
        saved, if any, depends on the hyperparameters. See docstring to
        save_predictions_and_summary(...).'''

        self._assert_version_is_ltl_or_lvl(version)

        # Lookup what to save based on (non-hash) hyperparameters.
        do_save_pred = self._do_save_pred(
            train_or_valid_str, version=version)
        do_save_summary = self._do_save_summary(
            train_or_valid_str, version=version)

        if not (do_save_pred or do_save_summary):
            return

        # Goal: only call predict() if absolutely necessary since it requires
        # substantial computation.
        do_generate_pred = (do_save_pred and pred is None) or \
            (do_save_summary and summary is None)

        # Summary always comes along when predictions are generated. So, make
        # sure we never have to call predict() just to get the summary if we
        # already got the predictions.
        if pred is not None:
            assert summary is not None, \
                'Summary must be provided if pred is provided.'

        if do_generate_pred:
            pred, summary = self.predict(data,
                train_or_valid_str=train_or_valid_str)

        if do_save_pred:
            self._save_pred(
                pred, train_or_valid_str, version=version)

        if do_save_summary:
            self._save_summary(
                summary, train_or_valid_str, version=version)

    def _do_save_pred(self, train_or_valid_str, version='lvl'):
        ''' Determines whether or not to save a set of predictions depending
        on hyperparameter settings.

        Returns: bool indicating whether or not to perform the save.
        '''
        if self.is_done:
            # Never use LTL model with validation data.
            # Accordingly, there is no hps.do_save_LTL_valid_predictions
            if train_or_valid_str == 'valid' and version == 'ltl':
                return False

            # E.g., do_save_lvl_train_predictions
            key = 'do_save_%s_%s_predictions' % (version, train_or_valid_str)
            return self.hps[key]
        else:
            return False

    def _do_save_summary(self, train_or_valid_str, version='lvl'):
        ''' Determines whether or not to save a summary of predictions
        depending on hyperparameter settings.

        Returns: bool indicating whether or not to perform the save.
        '''

        # Never use LTL model with validation data.
        # Accordingly, there is no hps.do_save_LTL_valid_summary
        if train_or_valid_str == 'valid' and version == 'ltl':
            return False

        # E.g., do_save_lvl_train_summary
        key = 'do_save_%s_%s_summary' % (version, train_or_valid_str)
        return self.hps[key]

    def _save_pred(self,
        predictions,
        train_or_valid_str,
        version='lvl',
        filetype=None):
        '''Saves all model predictions to disk.

        Args:
            predictions: dict containing model predictions.

            train_or_valid_str: either 'train' or 'valid', indicating whether
            data contains training data or validation data, respectively.

        Returns:
            None.
        '''
        if predictions is not None:

            print('\tSaving %s predictions (%s).' %
                (version, train_or_valid_str))

            self._save_pred_or_summary_helper(predictions,
                train_or_valid_str=train_or_valid_str,
                predictions_or_summary_str='predictions',
                version=version,
                filetype=filetype)

    def _save_summary(self,
        summary,
        train_or_valid_str,
        version='lvl',
        filetype='npz'):

        if summary is not None:

            print('\t\tSaving %s summary (%s).' %
                (str.upper(version), train_or_valid_str))

            self._save_pred_or_summary_helper(summary,
                train_or_valid_str=train_or_valid_str,
                predictions_or_summary_str='summary',
                version=version,
                filetype=filetype)

    def _save_done_file(self):
        '''Save an empty .done file (indicating that the training procedure
        ran to completion.

        Args:
            None.

        Returns:
            None.
        '''
        print('\tSaving .done file...')

        save_path = self._paths['done_path']
        file = open(save_path, 'wb')
        file.write('')
        file.close()

    def _save_pred_or_summary_helper(self,
        data_to_save,
        train_or_valid_str='train',
        predictions_or_summary_str='predictions',
        version='lvl',
        filetype='npz'):
        ''' Save data in a specified file format.

            Args:
                data_to_save: dict containing predictions or a summary thereof.

                filename: destination filename including extension.

            Returns:
                None.
        '''

        if filetype is None:
            if predictions_or_summary_str == 'predictions':
                filetype = self.hps.predictions_filetype
            elif predictions_or_summary_str == 'summary':
                filetype = self.hps.summary_filetype

        save_path = self._build_file_path(self.run_dir,
            train_or_valid_str=train_or_valid_str,
            predictions_or_summary_str=predictions_or_summary_str,
            version=version,
            filetype=filetype)

        if filetype == 'h5':
            self._save_h5(data_to_save, save_path)
        elif filetype == 'npz':
            self._save_npz(data_to_save, save_path)
        elif filetype == 'json':
            self._save_json(data_to_save, save_path)
        elif filetype == 'mat':
            self._save_mat(data_to_save, save_path)
        elif filetype == 'pkl':
            self._save_pkl(data_to_save, save_path)

    @classmethod
    def _load_pred_or_summary_helper(cls,
        run_dir,
        train_or_valid_str='train',
        predictions_or_summary_str='predictions',
        version='lvl',
        filetype='npz'):

        path_to_file = cls._build_file_path(run_dir,
            train_or_valid_str=train_or_valid_str,
            predictions_or_summary_str=predictions_or_summary_str,
            version=version,
            filetype='npz')

        if filetype == 'h5':
            return cls._load_h5(path_to_file)
        elif filetype == 'npz':
            return cls._load_npz(path_to_file)
        elif filetype == 'json':
            return cls._load_json(path_to_file)
        elif filetype == 'pkl':
            return cls._load_pkl(path_to_file)

    @staticmethod
    def _save_pkl(data_to_save, path_to_file):
        '''Pickle and save data as .pkl file.

        Args:
            data_to_save: any pickle-able object to be pickled and saved.

            path_to_file: path at which to save the data,
            including filename and extension.

        Returns:
            None.
        '''

        file = open(path_to_file, 'wb')
        file.write(cPickle.dumps(data_to_save))
        file.close()

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

    @staticmethod
    def _save_npz(data_to_save, path_to_file):
        '''Save data in Numpy .npz format.

        Args:
            data_to_save: Dict with values as numpy arrays or dicts that
            recursively satisfy this requirement (e.g., dict of numpy arrays).

            path_to_file: path at which to save the data,
            including filename and extension.

        Returns:
            None.
        '''
        flat_data = Hyperparameters.flatten(data_to_save)
        np.savez(path_to_file, **flat_data)

    @staticmethod
    def _load_npz(path_to_file):

        flat_data = dict(np.load(path_to_file, allow_pickle=True))
        data = Hyperparameters.unflatten(flat_data)
        return data

    @staticmethod
    def _save_mat(data_to_save, save_path):
        '''Save data as .mat file.

        Args:
            save_path: path at which to save the data, including filename and
            extension.

            data_to_save: dict containing data to be saved.

        Returns:
            None.
        '''

        spio.savemat(save_path, data_to_save)

    ''' Work in progress, largely untested:'''

    @staticmethod
    def _save_h5(data_to_save, path_to_file):
        '''Save data as HDF5 dataset.

        Args:
            data_to_save: Dict with values as numpy arrays or dicts that
            recursively satisfy this requirement (e.g., dict of numpy arrays).

            path_to_file: path at which to save the data,
            including filename and extension.

        Returns:
            None.
        '''
        flat_data = Hyperparameters.flatten(data_to_save)
        with h5py.File(path_to_file, 'w') as file:
            for key, val in flat_data.iteritems():
                assert '/' not in key, \
                    'data keys cannot contain \'/\': %s' % key
                file.create_dataset(key, data=val, compression=None)

    @staticmethod
    def _load_h5(data_to_save, path_to_file):

        with h5py.File(path_to_file, 'r') as file:
            for key, val in file.items():
                flat_data[key] = val
        data = Hyperparameters.unflatten(flat_data)
        return data

    @staticmethod
    def _save_json(data_to_save, path_to_file):
        '''Save data in JSON (.json) format.

        Args:
            data_to_save: Dict with values as python data types.

            path_to_file: path at which to save the data,
            including filename and extension.

        Returns:
            None.
        '''
        file = open(path_to_file, 'wb')
        json.dump(data_to_save, file)
        file.close()

    @staticmethod
    def _load_json(path_to_file):

        with open(path_to_file, 'r') as file:
            data = json.load(file)

        return data

    # Recently updated, largely untested.

    @classmethod
    def _exists_file(cls, run_dir,
        train_or_valid_str='train',
        predictions_or_summary_str='predictions',
        version='lvl',
        filetype='npz'):
        '''Checks if previously saved model predictions or summary exists.

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
            True if the file exists.
        '''

        path_to_file = cls._build_file_path(run_dir,
            train_or_valid_str=train_or_valid_str,
            predictions_or_summary_str=predictions_or_summary_str,
            version=version,
            filetype='npz')

        return os.path.exists(path_to_file)

    # *************************************************************************
    # Internal run directory management ***************************************
    # *************************************************************************

    @classmethod
    def _build_paths(cls, run_dir):
        '''Generates all paths relevant for saving and loading model data.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

        Returns:
            dict containing all paths relevant for saving and loading model
            data. Keys are strings, with suffixes '_dir' and '_path' referring
            to directories and filenames, respectively.
        '''

        subdirs = cls._build_subdirs(run_dir)
        hps_dir = cls._build_hps_dir(run_dir)
        seso_dir = subdirs['seso']
        ltl_dir = subdirs['ltl']
        lvl_dir = subdirs['lvl']
        events_dir = subdirs['events']

        file_paths = {
            'run_script_path': os.path.join(run_dir, 'run.sh'),

            'hps_path': cls._build_hps_path(run_dir, hps_dir=hps_dir),
            'hps_yaml_path': os.path.join(hps_dir, 'hyperparameters.yml'),

            'model_log_path': os.path.join(events_dir, 'model.log'),
            'loggers_log_path': os.path.join(events_dir, 'dependencies.log'),
            'done_path': cls._build_done_path(run_dir, events_dir=events_dir),

            'seso_ckpt_path': os.path.join(seso_dir, 'checkpoint.ckpt'),
            'ltl_ckpt_path': os.path.join(ltl_dir, 'ltl.ckpt'),
            'lvl_ckpt_path': os.path.join(lvl_dir, 'lvl.ckpt'),
            }

        return subdirs, file_paths

    @classmethod
    def _build_subdirs(cls, run_dir):

        D = {
            'hps': cls._build_hps_dir(run_dir),
            'seso': cls._build_seso_dir(run_dir),
            'ltl': cls._build_ltl_dir(run_dir),
            'lvl': cls._build_lvl_dir(run_dir),
            'events': cls._build_subdir(run_dir, 'events'),
            'fps': cls._build_subdir(run_dir, 'fps'),
        }

        return D

    @classmethod
    def _build_subdir(cls, run_dir, subdir):

        return os.path.join(run_dir, subdir)

    @classmethod
    def _build_hps_path(cls, run_dir, hps_dir=None):
        if hps_dir is None:
            hps_dir = cls._build_hps_dir(run_dir)
        return os.path.join(hps_dir, 'hyperparameters.pkl')

    @classmethod
    def _build_done_path(cls, run_dir, events_dir=None):
        if events_dir is None:
            events_dir = cls._build_subdir(run_dir, 'events')
        return os.path.join(events_dir, 'training.done')

    @classmethod
    def _build_file_path(cls, run_dir,
        train_or_valid_str='train',
        predictions_or_summary_str='predictions',
        version='lvl',
        filetype='npz'):
        ''' Builds paths to the various files that can be saved/loaded during
        and after training. This does not pertain to files created during
        directory setup and model construction.

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

        cls._assert_ckpt_version(version)
        cls._assert_filetype(filetype)

        assert predictions_or_summary_str in ['predictions', 'summary'],\
            ('Unsupported predictions_or_summary_str: %s' %
                predictions_or_summary_str)

        path_to_subdir = cls._build_subdir(run_dir, version)
        filename = '%s_%s.%s' % (
            train_or_valid_str,
            predictions_or_summary_str,
            filetype)
        path_to_file = os.path.join(path_to_subdir, filename)

        return path_to_file

    @classmethod
    def _build_hps_dir(cls, run_dir):
        return cls._build_subdir(run_dir, 'hps')

    @classmethod
    def _build_seso_dir(cls, run_dir):
        return cls._build_subdir(run_dir, 'seso')

    @classmethod
    def _build_ltl_dir(cls, run_dir):
        return cls._build_subdir(run_dir, 'ltl')

    @classmethod
    def _build_lvl_dir(cls, run_dir):
        return cls._build_subdir(run_dir, 'lvl')

    @classmethod
    def _build_fig_dir(cls, run_dir, version='seso'):
        '''
        'seso': /run_dir/figs/
        'ltl': /run_dir/ltl/figs/
        'lvl': /run_dir/lvl/figs/
        '''

        cls._assert_ckpt_version(version)
        if version == 'seso':
            subdir = run_dir
        else:
            subdir = cls._build_subdir(run_dir, version)
        return os.path.join(subdir, 'figs')

    # ************************************************************************
    # ************************************************************************
    # ************************************************************************