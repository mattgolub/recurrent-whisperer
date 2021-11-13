'''
RecurrentWhisperer.py
Written for Python 3.6.9 and TensorFlow 1.14
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

import sys
import os
import shutil
import logging
from copy import deepcopy
import subprocess
import warnings
import pdb

import tensorflow as tf
import numpy as np
import numpy.random as npr

# Imports for saving data, predictions, summaries
import pickle
import h5py, json, yaml
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
from EpochResults import EpochResults
from Timer import Timer

class RecurrentWhisperer(object):
    '''Base class for training recurrent neural networks or other deep
    learning models using TensorFlow. This class provides functionality for:

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
        _get_pred_ops(...)
        _build_data_feed_dict(...)
        _get_batch_size(...)
        _subselect_batch(...)

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

    def __init__(self, data_specs=None, **kwargs):
        '''Creates a RecurrentWhisperer object.

        Args:

            data_specs (optional): Any object.

                Contains data specifications that the model may need during
                construction, before a call to train() when the model first
                sees the training data (and possibly validation data).
                RecurrentWhisperer never looks inside data_specs, so the
                structure and contents are entirely up the the subclass that
                may use them.

                Importantly, data_specs are not placed into and saved by the
                Hyperparameters object. Thus, one can use data_specs for more
                cumbersome objects that don't play well with the
                Hyperparameters class (e.g., lists, dicts, etc). As a result,
                data_specs does not influence the Hyperparameters hash. Thus,
                the contents should be reproducible based on information in
                the subclass hyperparameters.

                One example use case is an object containing the sizes of
                various aspects of the data, which might be required for
                sizing the components of a model. One could also include a path
                to the data via _default_non_hash_hyperparameters(). Here, the
                data_specs would be entirely reproducible given the data path,
                so the hyperparameters would always uniquely specify the model.

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

                random_seed: non-negative int specifying the random seed for
                the numpy random generator used for randomly batching data and
                initializing model parameters. Set this to -1 to randomly
                generate the random_seed. Default: 0

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

                verbose: bool indicating whether to print additional detail
                during operation. Default: False.

                mode: string identifying the mode in which the model will be
                used. This is never used internally, and is only included for
                optional use by external run scripts. This is included here
                to simplify command-line argument parsing, which is already
                nicely handled by Hyperparameters.py Default: 'train'.

                do_batch_predictions: bool indicating whether to compute
                predictions in batches, or all in a single evaluation of the
                TF graph. This may be required for GPU/CPU memory management
                in large models or for large datasets. Default: False.

                do_train_mode_predict_on_train_data: bool indicating how to
                operate the model when running training data through a forward
                pass. Default: False.
                    True --> operate in "train mode", i.e., the same mode that
                    is used when computing gradient steps (forward+backward
                    pass). Train mode may include injecting noise or sampling
                    steps, which can act as a regularizer during training.
                    False --> operate in "predict mode", which typically has
                    noise sources turned off.
                 This is relevant for LTL predictions/summaries/visualization.
                 See kwarg: do_train_mode in predict() (and use it in your
                 implementation of _predict_batch(), if desired).

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

        self.timer = Timer(name='Total run time', do_retrospective=True)
        self.timer.start()

        if 'random_seed' in kwargs and kwargs['random_seed'] == -1:
            kwargs['random_seed'] = np.random.randint(2**31)

        hps = self.setup_hps(kwargs)
        self.timer.split('setup_hps')

        self.hps = hps
        self.data_specs = data_specs
        self.dtype = getattr(tf, hps.dtype)
        self._version = 'seso'
        self.prev_loss = None
        self.epoch_loss = None

        self._setup_run_dir()
        self.timer.split('_setup_run_dir')

        if hps.do_log_output:
            self._setup_logger()
            self.timer.split('_setup_logger')

        '''Make parameter initializations and data batching reproducible
        across runs.'''
        self.rng = npr.RandomState(hps.random_seed)
        tf.set_random_seed(hps.random_seed)
        ''' Note: Currently this state will not transfer across saves and
        restores. Thus behavior will only be reproducible for uninterrupted
        runs (i.e., that do not require restoring from a checkpoint). The fix
        would be to draw all random numbers needed for a run upon starting or
        restoring a run.'''
        self.timer.split('set_random_seed')

        self.adaptive_learning_rate = AdaptiveLearningRate(**hps.alr_hps)
        self.timer.split('init AdaptiveLearningRate')

        self.adaptive_grad_norm_clip = AdaptiveGradNormClip(**hps.agnc_hps)
        self.timer.split('init AdaptiveGradNormClip')

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
                self._setup_tensorboard()
                self.timer.split('_setup_tensorboard')

                self._setup_savers()
                self.timer.split('_setup_savers')

                self._setup_session()
                self.timer.split('_setup_session')

                if not hps.do_custom_restore:
                    self.initialize_or_restore()
                    self.print_trainable_variables()
                    self.timer.split('initialize_or_restore')

        print('')
        self.timer.print()
        print('')

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
            cls._default_rw_hash_hyperparameters(),
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
            cls._default_rw_non_hash_hyperparameters(),
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
            'verbose': False,

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
            'do_train_mode_predict_on_train_data': False,

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
            # Predictions and summary are never saved.
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

    @classmethod
    def _default_hash_hyperparameters(cls):
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
        raise Exception(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    @classmethod
    def _default_non_hash_hyperparameters(cls):
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
        raise Exception(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    @classmethod
    def parse_command_line(cls):
        ''' Parse command-line hyperparameter arguments (or arguments from
        higher-level shell script), and appropriately integrate them
        overriding default hyperparameters.

        Args:
            None.

        Returns:
            Dict of hyperparameters.
        '''
        default_hps = cls.default_hyperparameters()
        hps = Hyperparameters.parse_command_line(default_hps)
        return hps

    @classmethod
    def setup_hps(cls, hps_dict):

        return Hyperparameters(hps_dict,
            cls.default_hash_hyperparameters(),
            cls.default_non_hash_hyperparameters())

    @classmethod
    def get_command_line_call(cls,
        run_script,
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
        hp_names = list(flat_hps.keys())
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

                    # negative numbers misinterpreted by argparse as optional
                    # arg. This extra-space hack gets around it.
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
        subprocess.call(cmd_list)

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
            for d in list(subdirs.values()):
                os.makedirs(d)

            # Sub-subdirectories
            for version in ['seso', 'ltl', 'lvl']:
                d = self._build_fig_dir(run_dir, version=version)
                os.makedirs(d)

    def _setup_logger(self):
        '''Setup logging. Redirects (nearly) all printed output to the log
        file.

        Some output slips through the cracks, notably the output produced with
        calling tf.session

        Args:
            None.

        Returns:
            None.
        '''

        # Update all loggers that have been setup by dependencies
        # (e.g., Tensorflow)

        # Before changing where log messages go, make sure any currently in
        # the buffer get written...otherwise they may not refresh until
        # stdout is reset back to default, e.g. after the run terminates.
        sys.stdout.flush() # Attempt to shorten logging buffer time

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_level = logging.WARNING

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

    def _restore_logger_defaults(self):
        ''' Redirect all printing, errors, and warnings back to defaults. This
        undoes the logging redirecting enacted by _setup_logger().

        Args:
            None.

        Returns:
            None.
        '''
        self._log_file.close()
        sys.stdout = self._default_stdout
        sys.stderr = self._default_stderr

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

        self.device = '%s:%d' % (device_type, self.hps.device_id)
        print('Attempting to build TF model on %s\n' % self.device)

        ''' Some simple ops (e.g., tf.assign, tf.assign_add) must be placed on
        a CPU device. Instructing otherwise just ellicits warnings and
        overrides (at least in TF<=1.15).'''
        self.cpu_device = 'cpu:%d' % self.hps.cpu_device_id
        print('Placing CPU-only ops on %s\n' % self.cpu_device)

        if device_type == 'gpu':

            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                cuda_devices = ''

            print('\n\nCUDA_VISIBLE_DEVICES: %s' % cuda_devices)
            print('\n\n')
            print(subprocess.check_output(['nvidia-smi'],
                universal_newlines=True))
            print('\n\n')

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
        raise Exception(
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

            zipped_grads = list(zip(clipped_grads, vars_to_train))

            self.learning_rate = tf.placeholder(
                self.dtype, name='learning_rate')

            self.learning_rate_scale = tf.placeholder(
                self.dtype, name='learning_rate_scale')

            self.learning_rate_scaled = \
                self.learning_rate * self.learning_rate_scale

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate_scaled,
                **self.hps.adam_hps)

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
                self.hps())

        # Start training timer from scratch
        self.train_time_offset = 0.0

    # *************************************************************************
    # Tensorboard *************************************************************
    # *************************************************************************

    def _setup_tensorboard(self):
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

        if self.hps.do_save_tensorboard_histograms:
            self._setup_tensorboard_histograms()

    def _setup_tensorboard_summaries(self):
        '''Sets up Tensorboard summaries for monitoring the optimization.

        Args:
            None.

        Returns:
            None.
        '''
        self.tensorboard['merged_opt_summary'] = \
            self._build_merged_tensorboard_summaries(
                scope='tb-optimizer',
                ops_dict=self._get_tensorboard_summary_ops())

    def _get_tensorboard_summary_ops(self):
        ''' Returns a string-keyed dict of scalar TF ops to be logged by
        Tensorboard throughout the optimization.

        Args:
            None.

        Returns:
            Dict with strings as keys and scalar TF ops as values.
        '''
        return {
            self._loss_key: self.loss,
            self._grad_norm_key: self.grad_global_norm,
            'lvl': self.records['ops']['lvl'],
            'learning_rate': self.learning_rate,
            'learning_rate_scaled': self.learning_rate_scaled,
            'grad_norm_clip_val': self.grad_norm_clip_val,
            'clipped_grad_global_norm': self.clipped_grad_global_norm,
            'grad_clip_diff': self.clipped_grad_norm_diff
            }

    def _setup_tensorboard_histograms(self):
        '''Sets up Tensorboard histograms for monitoring all trainable
        variables throughout the optimization.

        Args:
            None.

        Returns:
            None.
        '''
        hist_ops = {}

        # Build string-keyed dict of trainable_variables
        for v in self.trainable_variables:
            hist_ops[v.name] = v

        self.tensorboard['merged_hist_summary'] = \
            self._build_merged_tensorboard_summaries(
                scope='model',
                ops_dict=hist_ops,
                summary_fcn=tf.summary.histogram)

    def _build_merged_tensorboard_summaries(self, scope, ops_dict,
        summary_fcn=tf.summary.scalar):
        ''' Builds and merges Tensorboard summaries.

        Args:
            scope: string for defining the scope the Tensorboard summaries to
            be created. This defines organizational structure within
            Tensorbaord.

            ops_dict: dictionary with string names as keys and TF objects as
            values. Names will be used as panel labels in Tensorboard.

            summary_fcn (optional): The Tensorflow summary function to be
            applied to TF objects in ops dict. Default: tf.summary_scalar

        Returns:
            A merged TF summary that, once executed via session.run(...), can
            be sent to Tensorboard via add_summary(...).
        '''

        summaries = []
        with tf.variable_scope(scope, reuse=False):
            for name, op in ops_dict.items():
                summaries.append(summary_fcn(name, op))

        return tf.summary.merge(summaries)

    def _update_train_tensorboard(self, feed_dict, ev_ops):
        ''' Updates Tensorboard based on a pass through a single-batch of
        training data.

        Args:
            feed_dict:

            ev_ops:

        Returns:
            None.
        '''
        if self.hps.do_save_tensorboard_summaries:

            ev_merged_opt_summary = ev_ops['merged_opt_summary']

            if self._epoch==0:
                '''Hack to prevent throwing the vertical axis on the
                Tensorboard figure for grad_norm_clip_val (grad_norm_clip val
                is initialized to an enormous number to prevent clipping
                before we know the scale of the gradients).'''
                feed_dict[self.grad_norm_clip_val] = np.nan
                ev_merged_opt_summary = \
                    self.session.run(
                        self.tensorboard['merged_opt_summary'],
                        feed_dict)

            self.tensorboard['writer'].add_summary(
                ev_merged_opt_summary, self._step)

        if self.hps.do_save_tensorboard_histograms:

            self.tensorboard['writer'].add_summary(
                ev_ops['merged_hist_summary'], self._step)

    def _update_valid_tensorboard(self, valid_summary):
        ''' Updates Tensorboard based on a pass through the validation data.

        Args:
            valid_summary: dict returned by predict().

        Returns:
            None.
        '''
        if self.hps.do_save_tensorboard_summaries:
            self._update_valid_tensorboard_summaries(valid_summary)

    def _update_valid_tensorboard_summaries(self, valid_summary):
        '''Updates the Tensorboard summaries corresponding to the validation
        data. Only called if do_save_tensorboard_summaries.

        Args:
            valid_summary: dict returned by predict().

        Returns:
            None.
        '''
        pass

    def _setup_tensorboard_images(self, figs=None):
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

        if figs is None:
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

        for fig_name, fig in figs.items():

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

        # Repeated calls will orphan an existing TF op :-(.
        images['merged_summaries'] = tf.summary.merge(images['summaries'])

        self.tensorboard['images'] = images

    def _update_tensorboard_images(self):
        ''' Imports figures into Tensorboard Images. Only called if:
                do_save_tensorboard_images and
                    (do_generate_training_visualizations or
                        do_generate_lvl_visualizations)

        Args:
            None.

        Returns:
            None.
        '''

        # Currently, cannot selectively update TB images. Update must be all
        # or none. This is because session.run(images['merged_summaries'], ...)
        # requires fed placeholders for all figs. To get around this would
        # require rebuilt images['merged_summaries'], where only the desired
        # figures' placeholder are merged. Or, the whole tf.summary.merge
        # could be sidestepped. Hopefully TF implemented the merge without
        # creating new ops (or at least new expensive ones). Otherwise the
        # former approach would waste GPU memory on redundant copies of figs.

        self._visualizations_timer.split('Tensorboard setup')
        print('\tUpdating Tensorboard images.')

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
           calling train(train_data, valid_data=None).
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

        if self._is_training_complete(self._ltl):
            # If restoring from a completed run, do not enter training loop
            # and do not save a new checkpoint.
            return

        self._maybe_generate_pretraining_visualizations(train_data, valid_data)

        # To do:
        # self._maybe_save_init_checkpoint()
        # -- Make sure to only save if self.epoch==0 (in case of restore)
        # -- This will encompass above visualizations
        # -- Make sure time is logged appropriately

        # Training loop
        print('Entering training loop.')

        done = False
        while not done:

            self._initialize_epoch()
            epoch_train_data = self._prepare_epoch_data(train_data)
            train_pred, train_summary = self._train_epoch(epoch_train_data)

            # The following may access epoch_train_data via self._epoch_results

            self._maybe_save_seso_checkpoint()
            self._maybe_save_ltl_checkpoint()
            self._maybe_save_lvl_checkpoint()
            self._maybe_update_visualizations(version='seso')
            done = self._is_training_complete()
            self._print_epoch_summary(train_summary)

        self.timer.split('train')

        self._maybe_save_final_seso_checkpoint()
        self._save_done_file()
        self._close_training(train_data, valid_data)
        self._print_run_summary()

    def _setup_training(self, train_data=None, valid_data=None):
        '''Performs any tasks that must be completed before entering the
        training loop in self.train().

        Args:
            train_data: dict containing the training data.

            valid_data: dict containing the validation data.

        Returns:
            None.
        '''

        self._has_valid_data = valid_data is not None

        # Use a helper class to organize predictions and prediction
        # summaries from this epoch. This initialization does not perform
        # any computation.
        self._epoch_results = EpochResults(
            model=self,
            train_data=train_data, # ok if this is None
            valid_data=valid_data, # ok if this is None
            do_batch=self.hps.do_batch_predictions,
            is_final=False)

        # Above is safe for the case of train_data=None (i.e., generating
        # train_data on-the-fly) because _prepare_epoch_data() updates
        # _epoch_results.train_data appropriately.

        self._initialize_epoch_timer()
        self.timer.split('_setup_training')

    def _initialize_epoch(self):

        self._print_epoch_state()
        self._epoch_results.reset()
        self._initialize_epoch_timer()

    def _initialize_epoch_timer(self):
        self._epoch_timer = Timer(name='Epoch', do_retrospective=True)
        self._epoch_timer.start()

    def _prepare_epoch_data(self, train_data):

        if train_data is None: # For on-the-fly data generation

            train_data = self.generate_data()

            self._epoch_results.train_data = train_data
            self._epoch_results.reset()

            self._epoch_timer.split('prep data')

        return train_data

    def _train_epoch(self, train_data=None):
        '''Performs training steps across an epoch of training data batches.

        Args:
            train_data: dict containing the training data. If not provided
            (i.e., train_data=None), data will be generated on-the-fly using
            generate_data(). Default: None.

        Returns:
            predictions: dict containing model predictions based on data. Key/
            value pairs will be specific to the subclass implementation.

            summary: dict containing high-level summaries of the predictions.
            Key/value pairs will be specific to the subclass implementation.
            Must contain key: 'loss' whose value is a scalar indicating the
            evaluation of the overall objective function being minimized
            during training.
        '''

        data_batches, batch_idxs = self._split_data_into_batches(train_data)
        self._epoch_timer.split('batching')

        pred_list = []
        summary_list = []

        n_batches = len(data_batches)
        for cnt, batch_data in enumerate(data_batches):

            if self.hps.verbose:

                batch_size = self._get_batch_size(batch_data)
                print('\tTraining on batch %d of %d (size=%d).' %
                    (cnt+1, n_batches, batch_size), end=' ')

                batch_timer = Timer(name='Batch')
                batch_timer.start()

            batch_pred, batch_summary = self._train_batch(batch_data)
            pred_list.append(batch_pred)
            summary_list.append(batch_summary)

            if self.hps.verbose:
                batch_timer.print_total_time()

        predictions, summary = self._combine_prediction_batches(
            pred_list, summary_list, batch_idxs)

        self._epoch_results.set(
            predictions=predictions,
            summary=summary,
            dataset='train',
            do_train_mode=True,)

        self.prev_loss = self.epoch_loss
        self.epoch_loss = self._get_summary_item(summary, self._loss_key)
        self.epoch_grad_norm = self._get_summary_item(
            summary, self._grad_norm_key)

        ''' Note, these updates are intentionally placed before any
        possible checkpointing for the epoch. This placement is critical
        for reproducible training trajectories when restoring (i.e., for
        robustness to unexpected restarts). See note in
        _print_run_summary(). '''
        self._update_learning_rate()
        self._update_grad_clipping()
        self._increment_epoch()

        self._epoch_timer.split('train')

        return predictions, summary

    def _train_batch(self, batch_data):
        '''Runs one training step. This function must evaluate the following:

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
        '''
        ops = {}

        # The forward-pass ops
        summary_ops = self._get_summary_ops()
        ops.update(summary_ops)

        pred_ops = self._get_pred_ops()
        ops.update(pred_ops)

        # The backward-pass ops
        train_ops = self._get_train_ops()
        ops.update(train_ops)

        feed_dict = self._build_feed_dict(batch_data, do_train_mode=True)
        ev_ops = self.session.run(ops, feed_dict=feed_dict)

        self._update_train_tensorboard(feed_dict, ev_ops)

        predictions = {}
        for key in pred_ops:
            predictions[key] = ev_ops[key]

        summary = {}
        for key in summary_ops:
            summary[key] = ev_ops[key]

        summary[self._grad_norm_key] = ev_ops[self._grad_norm_key]

        return predictions, summary

    def _get_train_ops(self):
        ''' Get the TF ops that result from a backward pass through the model.
        These are required for updating the model parameters (via SGD) and
        updating Tensorboard accordingly.

        Args:
            None.

        Returns:
            dict with (string label, TF ops) as (key, value) pairs.
        '''
        ops = {
            'train_op': self.train_op,
            self._grad_norm_key: self.grad_global_norm,
        }

        if self.hps.do_save_tensorboard_summaries:
            ops['merged_opt_summary'] = \
                self.tensorboard['merged_opt_summary']

        if self.hps.do_save_tensorboard_histograms:
            ops['merged_hist_summary'] = \
                self.tensorboard['merged_hist_summary']

        return ops

    def _get_summary_ops(self):
        # Don't include anything here that requires a backward pass through
        # the model (e.g., anything related to gradients)
        return {
            self._loss_key: self.loss,
            self._epoch_key: self._epoch_tf
            }

    def _build_feed_dict(self, data, do_train_mode=True):
        ''' Builds the feed dict needed to evaluate the model in either
        'train' or 'predict' mode.

        Args:
            data:

            do_train_mode:

        Returns:
            dict with (TF placeholder, feed value) as (key, value) pairs.
        '''

        feed_dict = {}
        data_feed_dict = self._build_data_feed_dict(data,
            do_train_mode=do_train_mode)
        feed_dict.update(data_feed_dict)

        if do_train_mode:
            optimizer_feed_dict = self._build_optimizer_feed_dict(
                learning_rate_scale=1.0)
            feed_dict.update(optimizer_feed_dict)

        return feed_dict

    def _build_optimizer_feed_dict(self, learning_rate_scale=1.0):
        ''' Build the feed_dict that provides the adaptive learning rate and
        adaptive gradient clipping parameters.

        Args:
            learning_rate_scale (optional): positive float that can be used to
            provide a batch-specific scaling of the learning rate (e.g., a
            function of batch size--see application note below).

        Returns:
            dict with (TF placeholder, feed value) as (key, value) pairs.
        '''


        ''' Application note:

        My typical usage had been: learning_rate_scale=np.sqrt(batch_size)
        However upon revisiting the literature, it seems a linear scaling
        may have more empirical justification (at least in feed-forward
        networks).

        "A bayesian perspective on generalization and stochastic gradient
        descent," by Smith & Le, ICLR 2018.

        "Don't decay the learning rate, increase the batch size"
        by Smith et al, ICLR 2018.
        (argues that equivalent performance can be achieved with fewer
        parameter updates by increasing batch size during training,
        while keeping learning rate constant-- all until batch size reaches
        ~10% of the dataset, at which point learning rate decay is
        recommended).

        "Control batch size and learning rate to generalize well:
        theoretical and empirical evidence", NeurIPS 2019.
        (argues for keeping a "not too large" ratio of batch size to
        learning rate).

        "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" by
        Goyal et al (Facebook). https://arxiv.org/pdf/1706.02677.pdf
        '''

        feed_dict = {
            self.learning_rate: self.adaptive_learning_rate(),
            self.learning_rate_scale: learning_rate_scale,
            self.grad_norm_clip_val: self.adaptive_grad_norm_clip()
        }

        return feed_dict

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
        raise Exception(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

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
        print('Epoch %d (step %d):' % (self._epoch+1, self._step+1))
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

    def _is_training_complete(self, epoch_loss=None):
        '''Determines whether the training optimization procedure should
        terminate. Termination criteria, governed by hyperparameters, are
        thresholds on the following:

            1) the training loss
            2) the learning rate
            3) the number of training epochs performed
            4) the number of training epochs performed since the lowest
               validation loss improved (only if do_check_lvl == True).

        Args:
            epoch_loss (optional):

        Returns:
            bool indicating whether any of the termination criteria have been
            met.
        '''

        hps = self.hps

        if epoch_loss is None:
            epoch_loss = self.epoch_loss

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

        self._epoch_timer.split('terminate', stop=True)

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

        train_data = self._prepare_epoch_data(train_data)

        self.save_final_results(train_data, valid_data, version='seso')
        self.save_final_results(train_data, valid_data, version='lvl')
        self.save_final_results(train_data, valid_data, version='ltl')

        if hps.do_log_output:
            self._restore_logger_defaults()

        self.timer.split('close_training')

    def save_final_results(self, train_data, valid_data, version='seso'):
        ''' Optionally save predictions and/or visualizations upon completion
        of training. This will optionally restore from 'ltl' or 'lvl'
        checkpoints, or will use the current model state if version is 'seso'.
        '''

        def do_predict(data, train_or_valid_str, version):

            if data is None:
                return False

            return self._do_save_pred(train_or_valid_str, version) or \
                self._do_save_summary(train_or_valid_str, version)

        hps = self.hps
        self._assert_ckpt_version(version)

        # Always want to compute new results, regardless of version, because
        # this is the only time predictions are generated with is_final=True.
        self._epoch_results = EpochResults(
            model=self,
            train_data=train_data,
            valid_data=valid_data,
            do_batch=hps.do_batch_predictions,
            is_final=True)

        for dataset in ['train', 'valid']:

            if dataset == 'train':
                data = train_data
                do_train_mode = hps.do_train_mode_predict_on_train_data
            elif dataset == 'valid':
                data = valid_data
                do_train_mode = False

            if do_predict(data, dataset, version):

                pred, summary = self._epoch_results.get(
                    dataset=dataset, do_train_mode=do_train_mode)

                # Always want to save this summary since it's the only one ever
                # computed using is_final=True (which can be used to trigger
                # one-time extensive summary metrics).
                self._save_summary(summary, dataset, version=version)

                if self._do_save_pred(dataset, version):
                    self._save_pred(pred, dataset, version=version)

        # Do not move this call! It leverages the results that have accumulated
        # in self._epoch_results from the loop above.
        self._save_final_visualizations(train_data, valid_data, version)

    def _save_final_visualizations(self, train_data, valid_data, version):

        hps = self.hps

        def do_train_vis(version, data):

            if data is None:
                return False

            if version == 'seso':
                return hps.do_generate_final_visualizations
            elif version == 'lvl':
                return False
            elif version == 'ltl':
                return hps.do_generate_ltl_visualizations

        def do_valid_vis(version, data):

            if data is None:
                return False

            if version == 'seso':
                return hps.do_generate_final_visualizations
            elif version == 'lvl':
                return hps.do_generate_lvl_visualizations
            elif version == 'ltl':
                return False

        def _do_save_visualizations(version):

            if version == 'seso':
                return hps.do_save_final_visualizations
            elif version == 'ltl':
                return hps.do_save_ltl_visualizations
            elif version == 'lvl':
                return hps.do_save_lvl_visualizations

        if do_train_vis(version, train_data):
            train_pred, train_summary = self._epoch_results.get(
                dataset='train',
                do_train_mode=hps.do_train_mode_predict_on_train_data)
        else:
            train_pred = train_summary = None

        if do_valid_vis(version, valid_data):
            valid_pred, valid_summary = self._epoch_results.get(
                dataset='valid', do_train_mode=False)
        else:
            valid_pred = valid_summary = None

        self.update_visualizations(
            train_data=train_data,
            train_pred=train_pred,
            train_summary=train_summary,
            valid_data=valid_data,
            valid_pred=valid_pred,
            valid_summary=valid_summary,
            version=version,
            do_save=_do_save_visualizations(version))

    def _assert_train_or_predict(self, train_or_predict_str):

        assert train_or_predict_str in ['train', 'predict'], \
            ('train_or_predict_str must be \'train\' or \'predict\', '
            'but was %s' % train_or_predict_str)

    # *************************************************************************
    # Prediction **************************************************************
    # *************************************************************************

    def predict(self, data,
        do_train_mode=False,
        do_batch=None,
        is_final=False):
        ''' Runs a forward pass through the model using given input data. If
        the input data are larger than the batch size, the data are processed
        sequentially in multiple batches.

        Args:
            data: dict containing requisite data for generating predictions.
            Key/value pairs will be specific to the subclass implementation.

            do_train_mode (optional): bool indicating whether run the forward
            pass in "train mode", i.e., the same mode that is used when
            computing gradient steps. E.g., train mode may include injecting
            noise or sampling steps, which can act as a regularizer during
            training. Default: False.

            do_batch (optional): bool indicating whether to split data into
            batches and then sequentially process those batches. This can be
            important for large models and/or large datasets relative to
            memory resources. Default: hps.do_batch_predictions.

            is_final (optional, advanced): bool indicating whether the model
            state is LTL, LVL, or similar. This option is not used in
            RecurrentWhisperer, but can be helpful in subclasses that may want
            customized predictions computed once training is complete.

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

        if do_train_mode:
            mode_str = ' (TRAIN MODE)'
        else:
            mode_str = ''

        if do_batch is None:
            do_batch = self.hps.do_batch_predictions

        if do_batch:

            batches_list, batch_indices = self._split_data_into_batches(data)
            n_batches = len(batches_list)
            pred_list = []
            summary_list = []
            for cnt, batch_data in enumerate(batches_list):

                batch_size = self._get_batch_size(batch_data)

                if self.hps.verbose:
                    print('\tPredict%s: batch %d of %d (%d trials)'
                          % (mode_str, cnt+1, n_batches, batch_size))

                batch_pred, batch_summary = self._predict_batch(batch_data,
                    do_train_mode=do_train_mode)

                pred_list.append(batch_pred)
                summary_list.append(batch_summary)

            predictions, summary = self._combine_prediction_batches(
                pred_list, summary_list, batch_indices)

        else:
            predictions, summary = self._predict_batch(data,
                do_train_mode=do_train_mode)

        return predictions, summary

    def _predict_batch(self, batch_data, do_train_mode=False):
        ''' Runs a forward pass through the model using a single batch of data.

        Args:
            batch_data: dict containing requisite data for generating
            predictions. Key/value pairs will be specific to the subclass
            implementation.

            do_train_mode (optional): bool indicating whether run the forward
            pass in "train mode", i.e., the same mode that is used when
            computing gradient steps. E.g., train mode may include injecting
            noise or sampling steps, which can act as a regularizer during
            training. Default: False.

        Returns:
            predictions: See docstring for predict().

            summary:  See docstring for predict().
        '''

        ops = {}

        pred_ops = self._get_pred_ops()
        ops.update(pred_ops)

        summary_ops = self._get_summary_ops()
        ops.update(summary_ops)

        feed_dict = self._build_data_feed_dict(batch_data,
            do_train_mode=do_train_mode)

        ev_ops = self.session.run(ops, feed_dict=feed_dict)

        predictions = {}
        for key in pred_ops:
            predictions[key] = ev_ops[key]

        summary = {}
        for key in summary_ops:
            summary[key] = ev_ops[key]

        assert (self._loss_key in summary),\
            ('summary must minimally contain key: '
            '\'%s\', but does not.' % self._loss_key)

        return predictions, summary

    def _get_pred_ops(self):
        ''' Get the dict of TF ops to be evaluated with each forward pass
        of the model. These are run by _predict_batch().

        Args:
            None.

        Returns:
            dict with (string label, TF ops) as (key, value) pairs.
        '''
        raise Exception(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    # *************************************************************************
    # Validation **************************************************************
    # *************************************************************************

    @property
    def _do_predict_validation(self):
        ''' Returns true if validation predictions or prediction summary
        are needed at the current epoch. '''

        return self._has_valid_data and \
            (self._do_update_validation or self._do_update_visualizations)

    @property
    def _do_update_validation(self):
        n = self.hps.n_epochs_per_lvl_update
        return np.mod(self._epoch, n) == 0

    # *************************************************************************
    # Data and batch management ***********************************************
    # *************************************************************************

    def generate_data(self):
        ''' Optionally generate data on-the-fly (e.g., during training), rather
        than relying on fixed sets of training and validation data. This is
        only called by train(...) when called using train_data=None.

        Args:
            None.

        Returns:
            data: dict containing the generated data.
        '''
        raise Exception(
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
        raise Exception(
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

            batch_indices: list, where each element, batch_indices[i], is a list
            of the trial indices for the corresponding batch of data in
            data_batches[i]. This is used to recombine the trials back into
            their original (i.e., pre-batching) order by
            _combine_prediction_batches().
        '''

        n_trials = self._get_batch_size(data)
        max_batch_size = self.hps.max_batch_size
        n_batches = int(np.ceil(float(n_trials)/max_batch_size))

        shuffled_indices = list(range(n_trials))
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

    @classmethod
    def _subselect_batch(cls, data, batch_idx):
        ''' Subselect a batch of data given the batch indices.

        Args:
            data: dict containing the to-be-subselected data.

            batch_idx: array-like of trial indices.

        Returns:
            subselected_data: dict containing the subselected data.
        '''

        raise Exception(
            '%s must be implemented by RecurrentWhisperer subclass'
             % sys._getframe().f_code.co_name)

    def _combine_prediction_batches(self,
        pred_list, summary_list, batch_indices):
        ''' Combines predictions and summaries across multiple batches. This is
        required by _train_epoch(...) and predict(...), which first split data
        into multiple batches before sequentially calling _train_batch(...) or
        _predict_batch(...), respectively, on each data batch.

        Args:
            pred_list: list of prediction dicts, each generated by
            _predict_batch(...)

            summary_list: list of summary dicts, each generated by
            _predict_batch(...).

            batch_indices: list of trial index lists, as returned by
            _split_data_into_batches(...).

        Returns:
            pred: a single prediction dict containing the combined predictions
            from pred_list.

            summary: a single summary dict containing the combined summaries
            from summary_list.
        '''

        raise Exception(
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
        for key in np.sort(list(batch_summaries[0].keys())):

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

    @classmethod
    def _get_summary_item(cls, summary, key):
        # Provided for ease of subclass reimplementation
        return summary[key]

    # *************************************************************************
    # Visualizations **********************************************************
    # *************************************************************************

    def update_visualizations(self,
        train_data=None,
        train_pred=None,
        train_summary=None,
        valid_data=None,
        valid_pred=None,
        valid_summary=None,
        version='seso',
        save_subdir=None,
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

        self.save_visualizations(
            do_save_figs=do_save,
            subdir=save_subdir,
            do_update_tensorboard=do_update_tensorboard,
            version=version)

        self._maybe_print_visualizations_timing()

    def save_visualizations(self,
        figs=None, # optionally pass in a subset of self.figs
        version='seso',
        subdir=None,
        do_save_figs=True,
        do_update_tensorboard=None):
        '''Saves individual figures to the relevant figure directory.

        Note: This is independent of Tensorboard Images.

        Args:
            figs (optional): Dict containing a subset of self.figs.

            version (optional): string indicating the state of the model used
            to generate the to-be-saved visualizations. Valid options are in
            list: _valid_ckpt_versions.

            subdir (optional): Enables advanced figure directories for
            subclasses. E.g., when stitching multiple datasets, dataset names
            can be used to create dataset-specific subdirectories via
            subdir=dataset_name. This option is never used internally to
            RecurrentWhisperer. Default: None.

            do_save_figs (optional): bool indicating whether to save
            individual figure files in the figure directory corresponding to
            version. Default: True.

            do_update_tensorboard (optional): bool indicating whether to save
            figures to tensorboard images. Default:
            hps.do_save_tensorboard_images.

        Returns:
            None.
        '''

        if do_update_tensorboard is None:
            do_update_tensorboard = self.hps.do_save_tensorboard_images

        if do_update_tensorboard:
            self._update_tensorboard_images()

        if do_save_figs:
            self._save_figs(figs=figs, version=version, subdir=subdir)

    def _update_visualizations(self,
        data, pred, train_or_valid_str, version):
        '''Updates visualizations in self.figs.

        Args:
            data: dict.

            pred: dict containing the result from predict(data).

            train_or_valid_str: either 'train' or 'valid', indicating whether
            data contains training data or validation data, respectively.

            version: string indicating the state of the model, which can be
            used to select which figures to generate. Valid options are in
            list: _valid_ckpt_versions.

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
        logged to the same Tensorboard image. Figure saving and Tensorboard
        logging are handled downstream of this function.
        '''

    def _maybe_generate_pretraining_visualizations(self,
        train_data, valid_data):

        # Visualizations generated from untrained network
        if self._do_pretraining_visualizations:

            train_data = self._prepare_epoch_data(train_data)

            train_pred, train_summary = self.predict(train_data,
                do_train_mode=self.hps.do_train_mode_predict_on_train_data,
                is_final=False)

            if valid_data is not None:
                valid_pred, valid_summary = self.predict(valid_data,
                    do_train_mode=False,
                    is_final=False)
            else:
                valid_pred = valid_summary = None

            self.update_visualizations(
                train_data=train_data,
                train_pred=train_pred,
                train_summary=train_summary,
                valid_data=valid_data,
                valid_pred=valid_pred,
                valid_summary=valid_summary,
                version='seso',
                do_save=self.hps.do_save_pretraining_visualizations)

            self.timer.split('init ckpt')

    def _maybe_update_visualizations(self, version='seso'):
        '''Updates visualizations if the current epoch number indicates that
        an update is due. Saves those visualization to Tensorboard or to
        individual figure files, depending on hyperparameters
        (do_save_tensorboard_images and do_save_training_visualizations,
        respectively.)

        Args:
            version:

        Returns:
            None.
        '''

        if self._do_update_training_visualizations:

            epoch_results = self._epoch_results
            train_data = epoch_results.train_data
            valid_data = epoch_results.valid_data

            do_save = self.hps.do_save_training_visualizations
            train_pred, train_summary = epoch_results.get(
                dataset='train',
                do_train_mode=self.hps.do_train_mode_predict_on_train_data)

            if valid_data is not None:
                valid_pred, valid_summary = epoch_results.get(
                    dataset='valid', do_train_mode=False)
            else:
                valid_pred = valid_summary = None

            self.update_visualizations(
                train_data=train_data,
                train_pred=train_pred,
                train_summary=train_summary,
                valid_data=valid_data,
                valid_pred=valid_pred,
                valid_summary=valid_summary,
                version=version,
                do_save=do_save)

        self._epoch_timer.split('visualize')

    def _save_figs(self,
        figs=None, # optionally pass in a subset of self.figs
        version='seso',
        subdir=None,):

        hps = self.hps
        fig_dir = self._build_fig_dir(self.run_dir,
            version=version,
            subdir=subdir)

        if figs is None:
            figs = self.figs

        print('\tSaving %s visualizations.' % version.upper())

        fig_names= list(figs.keys())
        fig_names.sort()

        for fig_name in fig_names:

            self._visualizations_timer.split('Saving: %s' % fig_name)

            file_path_no_ext = os.path.join(fig_dir, fig_name)
            figs_dir_i, filename_no_ext = os.path.split(file_path_no_ext)
            filename = filename_no_ext + '.' + hps.fig_filetype
            file_path = os.path.join(figs_dir_i, filename)

            # This fig's dir may have additional directory structure beyond
            # the already existing .../figs/ directory. Make it.
            if not os.path.isdir(figs_dir_i):
                os.makedirs(figs_dir_i)

            fig = figs[fig_name]
            fig.savefig(file_path,
                bbox_inches='tight',
                format=hps.fig_filetype,
                dpi=hps.fig_dpi)

        # Make sure whatever happens next doesn't affect timing of last save.
        self._visualizations_timer.split('Transition from saving.')

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
        return self._do_pretraining_visualizations or \
            self._do_update_training_visualizations

    @property
    def _do_pretraining_visualizations(self):

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

    @classmethod
    def refresh_figs(cls):
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

    @classmethod
    def get_hash_dir(cls, log_dir, run_hash):
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

        return cls._build_subdir(log_dir, run_hash)

    @classmethod
    def get_run_dir(cls, log_dir, run_hash, n_folds=None, fold_idx=None):
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

        hash_dir = cls.get_hash_dir(log_dir, run_hash)

        if (n_folds is not None) and (fold_idx is not None):

            fold_str = str('fold-%d-of-%d' % (fold_idx+1, n_folds))
            run_dir = cls._build_subdir(hash_dir, fold_str)

            return run_dir

        else:
            return hash_dir

    @classmethod
    def get_run_info(cls, run_dir):
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
        if cls.is_run_dir(run_dir):
            pass
        else:
            fold_names = list_dirs(run_dir)

            run_info = []
            for fold_name in fold_names:
                fold_dir = cls._build_subdir(run_dir, fold_name)
                if cls.is_run_dir(fold_dir):
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
        exists = [os.path.exists(d) for d in list(dirs.values())]
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

    @classmethod
    def get_hps_path(cls, run_dir, hps_dir=None):
        return cls._build_hps_path(run_dir, hps_dir=hps_dir)

    @classmethod
    def get_hps_mtime(cls, run_dir, hps_dir=None):
        hps_path = cls._build_hps_path(run_dir, hps_dir=hps_dir)
        return os.path.getmtime(hps_path)

    # *************************************************************************
    # Scalar access and updates ***********************************************
    # *************************************************************************

    _batch_size_key = 'batch_size'
    _loss_key = 'loss'
    _grad_norm_key = 'grad_global_norm'
    _epoch_key = 'epoch'

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
    def _epoch_tf(self):
        return self.records['ops']['epoch']

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
        return self.session.run(self._epoch_tf)

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
            placeholders[version]: self._format_loss(loss, version),
            placeholders[epoch_key]: epoch}
        ops = [
            update_ops[version],
            update_ops[epoch_key]
            ]

        self.session.run(ops, feed_dict=feed_dict)

    def _format_loss(self, loss, version):
        ''' Intermediary for formatting an LVL or LTL loss before it gets
        logged into records. Included for optional complexity that subclasses
        may require.
        '''
        return loss

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

        zipped_grads = list(zip(clipped_grads, vars_to_train))

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
    def load_hyperparameters(cls, run_dir, do_get_mtime=False):
        '''Load previously saved Hyperparameters.

        Args:
            run_dir: string containing the path to the directory where the
            model run was saved. See definition in __init__()

            do_get_mtime (optional): bool indicating whether or not to return
            the system file modification time for the Hyperparameters file.
            Default: False.

        Returns:

            hps_dict:
                dict containing the loaded hyperparameters.

            mtime (optional):
                float system file modification time for the hyperparameters
                file. Only returned if do_get_mtime is True.
        '''

        hps_path = cls._build_hps_path(run_dir)

        if os.path.exists(hps_path):
            hps_dict = Hyperparameters.restore(hps_path)
        else:
            raise IOError('%s not found.' % hps_path)

        if do_get_mtime:
            mtime = os.path.getmtime(hps_path)
            return hps_dict, mtime
        else:
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

            self._epoch_timer.split('seso')

    def _maybe_save_final_seso_checkpoint(self):
        ''' Saves the final SESO model checkpoint. This should only be called
        once, upon termination of the train() loop.

        Args:
            None.

        Returns:
            None.
        '''
        if self.hps.do_save_seso_ckpt:
            self._save_checkpoint(version='seso')

    def _maybe_save_ltl_checkpoint(self):
        ''' Saves a model checkpoint if the current training loss is lower than
        all previously evaluated training losses.

        Args:
            None.

        Returns:
            None.
        '''

        hps = self.hps
        version = 'ltl'
        train_loss = self.epoch_loss

        if self._do_save_ltl_checkpoint(train_loss):

            print('\tAchieved lowest training loss.')

            self._update_loss_records(train_loss, version=version)

            if hps.do_save_ltl_ckpt:
                self._save_checkpoint(version=version)

            if self.hps.do_save_ltl_train_summary:

                train_pred, train_summary = self._epoch_results.get(
                    dataset='train',
                    do_train_mode=hps.do_train_mode_predict_on_train_data)

                self._save_summary(train_summary, 'train', version=version)

        self._epoch_timer.split('ltl')

    def _maybe_save_lvl_checkpoint(self):
        ''' Runs a forward pass on the validation data, and saves a model
        checkpoint if the current validation loss is lower than all previously
        evaluated validation losses. Optionally, this will also generate and
        save model predictions over the training and validation data.

        Args:
            None.

        Returns:
            valid_pred, valid_summary:
                if using validation data and due for an update this epoch:
                    dicts as returned by predict(valid_data).
                otherwise:
                    Both are None.
        '''

        # ...if using validation data and due for an update this epoch
        if self._do_predict_validation:

            valid_pred, valid_summary = self._epoch_results.get(
                dataset='valid',
                do_train_mode=False)

            # ... if validation loss is better than previously seen
            hps = self.hps
            version = 'lvl'
            valid_loss = self._get_summary_item(valid_summary, self._loss_key)
            print('\tValidation loss: %.2e' % valid_loss)

            if self._do_save_lvl_checkpoint(valid_loss):

                print('\tAchieved lowest validation loss.')
                self._update_loss_records(valid_loss, version=version)

                if hps.do_save_lvl_ckpt:
                    self._save_checkpoint(version=version)

                self._maybe_save_pred_and_summary(
                    valid_pred, valid_summary, 'valid',
                    version=version,
                    is_final=False)

                if self._do_save_pred('train', version=version) or \
                    self._do_save_summary('train', version=version):

                    train_pred, train_summary = self._epoch_results.get(
                        dataset='train',
                        do_train_mode=hps.do_train_mode_predict_on_train_data)

                    self._maybe_save_pred_and_summary(
                        train_pred, train_summary, 'train',
                        version=version,
                        is_final=False)

            self._update_valid_tensorboard(valid_summary)

        else:
            valid_pred = valid_summary = None

        self._epoch_timer.split('lvl')

        return valid_pred, valid_summary

    ''' Currently there's a bit of asymmetry: ltl and lvl check
    hps.do_save_*_ckpt upstream, but hps.do_save_seso_ckpt is checked here.
    That's because LTL and LVL have more complicated checks and multiple tasks
    that depend on multiple checks. '''

    @property
    def _do_save_seso_checkpoint(self):
        n = self._epoch
        n_per_update = self.hps.n_epochs_per_seso_update
        return self.hps.do_save_seso_ckpt and np.mod(n, n_per_update) == 0

    def _do_save_ltl_checkpoint(self, train_loss):
        n = self._epoch
        n_next = self._epoch_next_ltl_check
        return n == 0 or (train_loss < self._ltl and n >= n_next)

    def _do_save_lvl_checkpoint(self, valid_loss):
        return self._epoch == 0 or valid_loss < self._lvl

    # *************************************************************************
    # *************************************************************************
    # *************************************************************************

    _valid_ckpt_versions = ['seso', 'ltl', 'lvl']

    def _save_checkpoint(self, version):
        '''Saves a model checkpoint, along with data for restoring the adaptive
        learning rate and the adaptive gradient clipper.

        Args:
            version: string indicating which version to label this checkpoint
            as. Valid options are in list: _valid_ckpt_versions.

        Returns:
            None.
        '''

        self._assert_ckpt_version(version)

        print('\tSaving %s checkpoint.' % version.upper())
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

    @classmethod
    def _assert_ckpt_version(cls, version):
        assert version in cls._valid_ckpt_versions, \
            'Unsupported version: %s' % str(version)

    @staticmethod
    def _assert_version_is_ltl_or_lvl(version):
        assert version in ['ltl', 'lvl'], \
            'Unsupported version: %s' % str(version)

    @classmethod
    def _assert_filetype(cls, filetype):
        assert filetype in cls._supported_filetypes,\
            'Unsupported filetype with extension: %s' % filetype

    # *************************************************************************
    # Restoring from model checkpoints ****************************************
    # *************************************************************************

    @classmethod
    def restore(cls, run_dir, version,
        data_specs=None,
        do_update_base_path=False):
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
        model = cls(data_specs=data_specs, **hps_dict)

        # Find and resotre parameters from lvl checkpoint
        model.restore_from_checkpoint(version, checkpoint_path=ckpt_path)

        return model

    def exists_checkpoint(self, version):
        '''
        Args:
            version: string indicating which version to label this checkpoint
            as. Valid options are in list: _valid_ckpt_versions.
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

        self._version = version.lower()

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
    def get_train_summary_mtime(cls, run_dir, version='lvl', filtetype='npz'):

        summary_path = cls._build_file_path(run_dir,
            train_or_valid_str='train',
            predictions_or_summary_str='summary',
            version=version,
            filetype=filetype)

        return os.path.getmtime(summary_path)

    @classmethod
    def get_vaild_summary_mtime(cls, run_dir, version='lvl', filtetype='npz'):

        summary_path = cls._build_file_path(run_dir,
            train_or_valid_str='valid',
            predictions_or_summary_str='summary',
            version=version,
            filetype=filetype)

        return os.path.getmtime(summary_path)

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

    def save_predictions_and_summary(self, data, train_or_valid_str, version,
        do_train_mode=False,
        do_batch_predictions=None,
        is_final=True,
        predictions_filetype=None,
        summary_filetype=None):
        ''' Saves model predictions and a prediction summary, regardless of the
        hyperparameters. This is provided for external convenience, and is
        never used internally.

        Args:
            data: dict containing the data over which predictions are
            generated. This can be the training data or the validation data.

            train_or_valid_str: either 'train' or 'valid', indicating whether
            data contains training data or validation data, respectively.
            The resulting filenames will reflect this.

            version: 'ltl', 'lvl', or 'seso' indicating whether the state of
            the model is lowest-training-loss, lowest-validation-loss, or
            'save-every-so-often', respectively. This determines the names and
            locations of the files to be saved.

        Returns:
            None.
        '''

        self._assert_ckpt_version(version)

        pred, summary = self.predict(data,
            do_train_mode=do_train_mode,
            do_batch=do_batch_predictions,
            is_final=is_final)

        self._save_pred(pred, train_or_valid_str,
            version=version,
            filetype=predictions_filetype)

        self._save_summary(summary, train_or_valid_str,
            version=version,
            filetype=summary_filetype)

    def save_summary(self, data, train_or_valid_str, version,
        do_train_mode=False,
        do_batch_predictions=None,
        is_final=True,
        filetype=None):
        ''' Saves model prediction summary without saving the (bulky)
        predictions themselves. This save is done regardless of the
        hyperparameters (which could otherwise indicate that no summaries are
        to be saved during training). This is provided for external
        convenience, and is never used internally.

        Args:
            data: dict containing the data over which predictions are
            generated. This can be the training data or the validation data.

            train_or_valid_str: either 'train' or 'valid', indicating whether
            data contains training data or validation data, respectively.
            The resulting filenames will reflect this.

            version: 'ltl', 'lvl', or 'seso' indicating whether the state of
            the model is lowest-training-loss, lowest-validation-loss, or
            'save-every-so-often', respectively. This determines the names and
            locations of the files to be saved.

        Returns:
            None.
        '''

        self._assert_ckpt_version(version)

        pred, summary = self.predict(data,
            do_train_mode=do_train_mode,
            do_batch=do_batch_predictions,
            is_final=is_final)

        self._save_summary(summary, train_or_valid_str,
            version=version,
            filetype=filetype)

    @classmethod
    def load_train_predictions(cls, run_dir,
        do_get_mtime=False,
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
            do_get_mtime=do_get_mtime,
            version=version,
            filetype=filetype)

    @classmethod
    def load_train_summary(cls, run_dir,
        do_get_mtime=False,
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
            do_get_mtime=do_get_mtime,
            version=version,
            filetype=filetype)

    @classmethod
    def load_valid_predictions(cls, run_dir,
        do_get_mtime=False,
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
            do_get_mtime=do_get_mtime,
            version=version,
            filetype=filetype)

    @classmethod
    def load_valid_summary(cls, run_dir,
        do_get_mtime=False,
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
            do_get_mtime=do_get_mtime,
            version=version,
            filetype=filetype)

    def _maybe_save_pred_and_summary(self, pred, summary, train_or_valid_str,
        do_train_mode=False,
        version='lvl',
        is_final=False):
        '''Saves model predictions and/or a prediction summary. Which are
        saved, if any, depends on the hyperparameters. See docstring to
        save_predictions_and_summary(...).'''

        self._assert_version_is_ltl_or_lvl(version)

        if self._do_save_pred(train_or_valid_str, version=version):
            self._save_pred(pred, train_or_valid_str, version=version)

        if self._do_save_summary(train_or_valid_str, version=version):
            self._save_summary(summary, train_or_valid_str, version=version)

    def _do_save_pred(self, train_or_valid_str, version='lvl'):
        ''' Determines whether or not to save a set of predictions depending
        on hyperparameter settings.

        Returns: bool indicating whether or not to perform the save.
        '''
        if self.is_done(self.run_dir):

            if version == 'seso':
                return False

            # Never use LTL model with validation data.
            # Accordingly, there is no hps.do_save_ltl_valid_predictions
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
        if version == 'seso':
            return False

        # Never use LTL model with validation data.
        # Accordingly, there is no hps.do_save_ltl_valid_summary
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
                (version.upper(), train_or_valid_str))

            self._save_pred_or_summary_helper(predictions,
                train_or_valid_str=train_or_valid_str,
                predictions_or_summary_str='predictions',
                version=version,
                filetype=filetype)

    def _save_summary(self,
        summary,
        train_or_valid_str,
        version='lvl',
        filetype=None):

        if summary is not None:

            print('\tSaving %s summary (%s).' %
                (version.upper(), train_or_valid_str))

            self._save_pred_or_summary_helper(summary,
                train_or_valid_str=train_or_valid_str,
                predictions_or_summary_str='summary',
                version=version,
                filetype=filetype)

    def _save_done_file(self):
        '''Save .done file (an empty file whose existence indicates that the
        training procedure ran to self termination.

        CRITICAL: This must be called after saving final SESO checkpoint,
        but before doing a bunch of other stuff that might fail. This way,
        if any of that stuff does fail, the .done file will be present,
        indicating safe to interpret checkpoint model as final. Also, subclass
        implementations of predict(), update_visualizations(), etc can check
        self.is_done to do certain expensive things just once at the end of
        training, rather than on every call throughout training.

        Args:
            None.

        Returns:
            None.
        '''
        print('\tSaving .done file.')

        save_path = self._paths['done_path']
        file = open(save_path, 'w')
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
        elif filetype == 'mat':
            self._save_mat(data_to_save, save_path)
        elif filetype == 'pkl':
            self._save_pkl(data_to_save, save_path)
        elif filetype in ['json', 'yaml']:

            # Only supported for summary (not predictions)
            if predictions_or_summary_str != 'summary':
                raise ValueError(
                    'Saving predictions as %s is not supported.' %
                        filetype)

            json_data = self._jsonify(data_to_save)
            if filetype == 'json':
                self._save_json(json_data, save_path)
            elif filetype == 'yaml':
                # This is still problematic, with platform-specific issues.
                warnings.warn('Caution: Saving summary as yaml '
                    'can yeild unpredictable results.')
                self._save_yaml(json_data, save_path)

    @classmethod
    def _load_pred_or_summary_helper(cls,
        run_dir,
        train_or_valid_str='train',
        predictions_or_summary_str='predictions',
        do_get_mtime=False,
        version='lvl',
        filetype='npz'):

        path_to_file = cls._build_file_path(run_dir,
            train_or_valid_str=train_or_valid_str,
            predictions_or_summary_str=predictions_or_summary_str,
            version=version,
            filetype='npz')

        if filetype == 'h5':
            result = cls._load_h5(path_to_file)
        elif filetype == 'npz':
            result = cls._load_npz(path_to_file)
        elif filetype == 'json':
            result = cls._load_json(path_to_file)
        elif filetype == 'pkl':
            result = cls._load_pkl(path_to_file)

        if do_get_mtime:
            mtime = os.path.getmtime(path_to_file)
            return result, mtime
        else:
            return result

    _supported_filetypes = ['h5', 'npz', 'json', 'mat', 'pkl', 'yaml']

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
        file.write(pickle.dumps(data_to_save))
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
            data = pickle.loads(load_path)
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

    @classmethod
    def _jsonify(cls, D):
        ''' Creates a deep copy of a dict that is safe for saving as JSON.

        Args:
            D: python dict with all keys as strings or dicts that recursively
            satisfy this requirement.

        Returns:
            Dict with all representations safe for saving as JSON.
        '''
        def isnumpy(val):

            return type(val).__module__ == np.__name__

        def jsonify_numpy(val):
            ''' Converts a Numpy object into a JSON-safe Python-type
            representation. Numpy scalars are converted to int or float types.
            Numpy arrays are converted to lists.

            Args:
                val: any Numpy object (e.g., scalar, array).

            Returns:
                JSON-safe representation of val.
            '''
            if isinstance(val, np.integer):
                return int(val)
            elif isinstance(val, np.floating):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            else:
                raise TypeError('Unsupported type(val)=%s.' % str(type(val)))

        json_dict = {}

        for key, val in D.items():

            if val is None:
                json_dict[key] = 'None'
            elif isnumpy(val):
                json_dict[key] = jsonify_numpy(val)
            elif isinstance (val, dict):
                json_dict[key] = cls._jsonify(val)
            elif isinstance(val, str):
                # No conversion necessary. Just removing this possibility from
                # the catch-all below for more informative error reporting.
                # (strings  are safe for JSON, so they won't relate to errors)
                json_dict[key] = val # just shallow copy non Numpy types
            else:
                print('_jsonify() encountered unsupported datatype: '
                    'summary[\'%s\'] = %s (%s)' %
                    (key, str(val), str(type(val))))

                json_dict[key] = val # just shallow copy, what else can you do?

        return json_dict

    @classmethod
    def _print_dict_types(cls, D, n_indent=0):
        ''' Print the datatypes of each element of a python Dict.
        Helpful for debugging encoding issues when saving train/valid
        summary dicts as .json or .yaml.
        '''

        sorted_keys = list(D.keys())
        sorted_keys.sort()

        for key in sorted_keys:

            val = D[key]

            if isinstance(val, dict):
                cls.print_dict_types(val, n_indent=n_indent+1)
            else:
                indent_str = n_indent * '\t'
                print('%s%s: %s' % (indent_str, str(type(val)), key))

    @classmethod
    def _save_yaml(cls, data_to_save, path_to_file):
        '''Save data in YAML format.

        Args:
            data_to_save: Dict with values python data types or dicts that
            recursively satisfy this requirement (e.g., dict of python types).

            path_to_file: path at which to save the data,
            including filename and extension.

        Returns:
            None.
        '''
        with open(path_to_file, 'w') as yaml_file:
            yaml.dump(data_to_save, yaml_file,
                default_flow_style=False,
                canonical=False)

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
            for key, val in flat_data.items():
                assert '/' not in key, \
                    'data keys cannot contain \'/\': %s' % key
                file.create_dataset(key, data=val, compression=None)

    @staticmethod
    def _load_h5(data_to_save, path_to_file):

        with h5py.File(path_to_file, 'r') as file:
            for key, val in list(file.items()):
                flat_data[key] = val
        data = Hyperparameters.unflatten(flat_data)
        return data

    @classmethod
    def _save_json(cls, data_to_save, path_to_file):
        '''Save data in JSON (.json) format.

        Args:
            data_to_save: Dict with values python data types or dicts that
            recursively satisfy this requirement (e.g., dict of python types).

            path_to_file: path at which to save the data,
            including filename and extension.

        Returns:
            None.
        '''
        file = open(path_to_file, 'wb')
        json.dump(data_to_save, file, indent=4)
        file.close()

    @staticmethod
    def _load_json(path_to_file):

        # To do: "Decode" lists back into numpy arrays.
        with open(path_to_file, 'r') as file:
            data = json.load(file)

        return data

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
    def _build_fig_dir(cls, run_dir, version='seso', subdir=None):
        ''' Builds string path to a figures directory.

        version='seso' --> '/<run_dir>/figs/'
        version='ltl'  --> '/<run_dir>/ltl/figs/'
        version='lvl'  --> '/<run_dir>/lvl/figs/'

        Args:
            run_dir:

            version:

            subdir (optional): Enables advanced figure directories for
            subclasses, e.g., when stitching multiple datasets, can append
            dataset name via subdir=dataset_name. This option is never used
            internally to RecurrentWhisperer. Default: None.
        '''

        cls._assert_ckpt_version(version)
        if version == 'seso':
            version_dir = run_dir
        else:
            version_dir = cls._build_subdir(run_dir, version)

        fig_dir = cls._build_subdir(version_dir, 'figs')

        if subdir is None:
            return fig_dir
        else:
            return cls._build_subdir(fig_dir, subdir)

    # ************************************************************************
    # ************************************************************************
    # ************************************************************************