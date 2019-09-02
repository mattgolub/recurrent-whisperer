'''
AdaptiveLearningRate.py
Version 1.0
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''
import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import cPickle

class AdaptiveLearningRate(object):
	'''Class for managing an adaptive learning rate schedule based on the
	recent history of loss values. The adaptive schedule begins with an
	optional warm-up period, during which the learning rate logarithmically
	increases up to the initial rate. For the remainder of the training
	procedure, the learning rate will increase following a period of monotonic
	improvements in the loss and will decrease if a loss is encountered that
	is worse than all losses in the recent period. Hyperparameters control the
	length of each of these periods and the extent of each type of learning
	rate change.

	Note that this control flow is asymmetric--stricter criteria must be met
	for increases than for decreases in the learning rate This choice 1)
	encourages decreases in the learning rate when moving into regimes with a
	flat loss surface, and 2) attempts to avoid instabilities that can arise
	when the learning rate is too high (and the often irreversible
	pathological parameter updates that can result). Practically,
	hyperparameters may need to be tuned to optimize the learning schedule and
	to ensure that the learning rate does not explode.

	See test(...) to simulate learning rate trajectories based on specified
	hyperparameters.

	The standard usage is as follows:

	# Set hyperparameters as desired.
	alr_hps = dict()
	alr_hps['initial_rate'] = 1.0
	alr_hps['n_warmup_steps'] = 0
	alr_hps['warmup_scale'] = 0.001
	alr_hps['do_decrease_rate'] = True
	alr_hps['min_steps_per_decrease'] = 5
	alr_hps['decrease_factor'] = 0.95
	alr_hps['do_increase_rate'] = True
	alr_hps['min_steps_per_increase'] = 5
	alr_hps['increase_factor'] = 1./0.95
	alr_hps['verbose'] = False
	alr = AdaptiveLearningRate(**alr_hps)

	while some_conditions(...):
		# This loop defines one step of the training procedure.

		# Get the current learning rate
		learning_rate = alr()

		# Use the current learning rate to update the model parameters.
		# Get the loss of the model after the update.
		params, loss = run_one_training_step(params, learning_rate, ...)

		# Update the learning rate based on the most recent loss value
		# and an internally managed history of loss values.
		alr.update(loss)

		# (Optional): Occasionally save model checkpoints along with the
		# AdaptiveLearningRate object (for seamless restoration of a training
		# session)
		if some_other_conditions(...):
			save_checkpoint(params, ...)
			alr.save(...)
	'''

	# Default hyperparameter settings (see comments in __init__)
	_INITIAL_RATE = 1.0
	_N_WARMUP_STEPS = 0
	_WARMUP_SCALE = 0.001
	_DO_DECREASE_RATE = True
	_MIN_STEPS_PER_DECREASE = 5
	_DECREASE_FACTOR = 0.95
	_DO_INCREASE_RATE = True
	_MIN_STEPS_PER_INCREASE = 5
	_INCREASE_FACTOR = 1./0.95
	_VERBOSE = False

	def __init__(self,
		initial_rate = _INITIAL_RATE,
		n_warmup_steps = _N_WARMUP_STEPS,
		warmup_scale = _WARMUP_SCALE,
		do_decrease_rate = _DO_DECREASE_RATE,
		min_steps_per_decrease = _MIN_STEPS_PER_DECREASE,
		decrease_factor = _DECREASE_FACTOR,
		do_increase_rate = _DO_INCREASE_RATE,
		min_steps_per_increase = _MIN_STEPS_PER_INCREASE,
		increase_factor = _INCREASE_FACTOR,
		verbose = _VERBOSE):
		'''Builds an AdaptiveLearningRate object

		Args:
			A set of optional keyword arguments for overriding the default values of the following hyperparameters:

			initial_rate: A non-negative float specifying the initial learning
			rate. Default: 1.0.

			n_warmup_steps: A non-negative int specifying the number of warm-up
			steps to take. During these warm-up steps, the learning rate will
			logarithmically increase up to initial_rate. Default: 0 (i.e., no
			warm-up).

			warmup_scale: A float between 0 and 1 specifying the learning rate
			on the first warm-up step, relative to initial_rate. The first
			warm-up learning rate is warmup_scale * initial_rate. Default:
			0.001.

			do_decrease_rate: A bool indicating whether or not to decrease the
			learning rate during training (after any warm-up). Default: True.

			min_steps_per_decrease: A non-negative int specifying the number
			of recent steps' loss values to consider when deciding whether to
			decrease the learning rate. Learning rate decreases are made when
			a loss value is encountered that is worse than every loss value in
			this window. When the learning rate is decreased, no further
			decreases are considered until this many new steps have
			transpired. Larger values will slow convergence due to the
			learning rate. Default 5.

			decrease_factor: A float between 0 and 1 specifying the extent of
			learning rate decreases. Whenever a decrease is made, the learning
			rate decreases from x to decrease_factor * x. Values closer to 1
			will slow convergence due to the learning rate. Default: 0.95.

			do_increase_rate: A bool indicating whether or not to increase the
			learning rate during training (after any warm-up). Default: True.

			min_steps_per_increase: A non-negative int specifying the number
			of recent steps' loss values to consider when deciding whether to
			increase the learning rate. Learning rate increases are made when
			the loss has monotonically increased over this many steps. When
			the learning rate is increased, no further increases are
			considered until this many new steps have transpired. Default 5.

			increase_factor: A float greater than 1 specifying the extent of
			learning rate increases. Whenever an increase is made, the
			learning rate increases from x to increase_factor * x. Larger
			values will slow convergence due to the learning rate. Default:
			1./0.95.

			verbose: A bool indicating whether or not to print status updates.
			False.
		'''

		self.step = 0
		self.step_last_update = -1
		self.prev_rate = None
		self.loss_log = []

		self.initial_rate = initial_rate
		self.do_decrease_rate = do_decrease_rate
		self.decrease_factor = decrease_factor
		self.min_steps_per_decrease = min_steps_per_decrease
		self.do_increase_rate = do_increase_rate
		self.increase_factor = increase_factor
		self.min_steps_per_increase = min_steps_per_increase
		self.n_warmup_steps = n_warmup_steps
		self.warmup_scale = warmup_scale

		self.save_filename = 'learn_rate.pkl'

		self._validate_hyperparameters()

		self.warmup_rates = self._get_warmup_rates()

		self.verbose = verbose

		if n_warmup_steps > 0:
			self.learning_rate = self.warmup_rates[0]
		else:
			self.learning_rate = initial_rate

	def _validate_hyperparameters(self):
		'''Checks that critical hyperparameters have valid values.

		Args:
			None.

		Returns:
			None.

		Raises:
			Various ValueErrors depending on the violating hyperparameter(s).
		'''
		def assert_non_negative(attr_str):
			'''
			Args:
				attr_str: The name of a class variable.

			Returns:
				None.

			Raises:
				ValueError('%s must be non-negative but was %d' % (...))
			'''
			val = getattr(self, attr_str)
			if val < 0:
				raise ValueError('%s must be non-negative but was %d'
								 % (attr_str, val))

		assert_non_negative('initial_rate')
		assert_non_negative('n_warmup_steps')
		assert_non_negative('min_steps_per_decrease')
		assert_non_negative('min_steps_per_increase')

		if self.decrease_factor > 1.0 or self.decrease_factor < 0.:
			raise ValueError('decrease_factor must be between 0 and 1, but was %f'
							 % self.decrease_factor)

		if self.increase_factor < 1.0:
			raise ValueError('increase_factor must be >= 1, but was %f'
							 % self.increase_factor)

	def __call__(self):
		'''Returns the current learning rate.'''

		return self.learning_rate

	def update(self, loss):
		'''Updates the learning rate based on the most recent loss value
		relative to the recent history of loss values.

		Args:
			loss: A float indicating the loss from the current training step.

		Returns:
			A float indicating the updated learning rate.
		'''
		self.loss_log.append(loss)

		step = self.step
		cur_rate = self.learning_rate
		step_last_update = self.step_last_update

		self.prev_rate = cur_rate

		if step <= self.n_warmup_steps:
			'''If step indicates that we are still in the warm-up, the new rate is determined entirely based on the warm-up schedule.'''
			if step < self.n_warmup_steps:
				self.learning_rate = self.warmup_rates[step]
				if self.verbose:
					print('Warm-up (%d of %d): Learning rate set to %.2e'
						  % (step+1,self.n_warmup_steps,self.learning_rate))
			else: # step == n_warmup_steps:
				self.learning_rate = self.initial_rate
				if self.verbose:
					print('Warm-up complete (or no warm-up). Learning rate set to %.2e'
						  % self.learning_rate)
			self.step_last_update = step

			'''Otherwise, rate may be kept, increased, or decreased based on
			recent loss history.'''
		elif self._conditional_decrease_rate():
			self.step_last_update = step
		elif self._conditional_increase_rate():
			self.step_last_update = step

		self.step = step + 1

		return self.learning_rate

	def _get_warmup_rates(self):
		'''Determines the warm-up schedule of logarithmically increasing
		learning rates, culminating at the desired initial rate.

		Args:
			None.

		Returns:
			An [n_warmup_steps,] numpy array containing the learning rates for
			each step of the warm-up period.

		'''
		scale = self.warmup_scale
		warmup_start = scale*self.initial_rate
		warmup_stop = self.initial_rate
		n = self.n_warmup_steps + 1
		warmup_rates = np.logspace(np.log10(warmup_start),
								   np.log10(warmup_stop),
								   n)
		return warmup_rates

	def _conditional_increase_rate(self):
		'''Increases the learning rate if loss values have monotonically
		decreased over the past n steps, and if no learning rate changes have
		been made in the last n steps, where n=min_steps_per_increase.

		Args:
			None.

		Returns:
			A bool indicating whether the learning rate was increased.
		'''

		did_increase_rate = False
		n = self.min_steps_per_increase

		if self.do_increase_rate and self.step>=(self.step_last_update + n):

			batch_loss_window = self.loss_log[-(1+n):]
			lastBatchLoss = batch_loss_window[-1]

			if all(np.less(batch_loss_window[1:],batch_loss_window[:-1])):
				self.learning_rate = self.learning_rate * self.increase_factor
				did_increase_rate = True
				if self.verbose:
					print('Learning rate increased to %.2e'
						  % self.learning_rate)

		return did_increase_rate

	def _conditional_decrease_rate(self):
		'''Decreases the learning rate if the most recent loss is worse than
		all of the previous n loss values, and if no learning rate changes
		have been made in the last n steps, where n=min_steps_per_decrease.

		Args:
			None.

		Returns:
			A bool indicating whether the learning rate was decreased.
		'''

		did_decrease_rate = False
		n = self.min_steps_per_decrease

		if self.do_decrease_rate and self.step>=(self.step_last_update + n):

			batch_loss_window = self.loss_log[-(1+n):]
			lastBatchLoss = batch_loss_window[-1]

			if all(np.greater(batch_loss_window[-1],batch_loss_window[:-1])):
				self.learning_rate = self.learning_rate * self.decrease_factor
				did_decrease_rate = True
				if self.verbose:
					print('Learning rate decreased to %.2e'
						  % self.learning_rate)

		return did_decrease_rate

	def save(self, save_dir):
		'''Saves the current state of the AdaptiveLearningRate object.

		Args:
			save_dir: A string containing the directory in which to save.

		Returns:
			None.
		'''
		if self.verbose:
			print('Saving AdaptiveLearningRate.')
		save_path = os.path.join(save_dir, self.save_filename)
		file = open(save_path,'w')
		file.write(cPickle.dumps(self.__dict__))
		file.close

	def restore(self, restore_dir):
		'''Restores the state of a previously saved AdaptiveLearningRate
		object.

		Args:
			restore_dir: A string containing the directory in which to find a
			previously saved AdaptiveLearningRate object.

		Returns:
			None.
		'''
		if self.verbose:
			print('Restoring AdaptiveLearningRate.')
		restore_path = os.path.join(restore_dir, self.save_filename)
		file = open(restore_path,'r')
		restore_data = file.read()
		file.close()
		self.__dict__ = cPickle.loads(restore_data)

def test(n_steps=1000, bias=0., **kwargs):
	''' Generates and plots an adaptive learning rate schedule based on a loss
	function that is a 1-dimensional biased random walk. This can be used as a
	zero-th order analysis of hyperparameter settings, understanding that in a
	realistic optimization setting, the loss will depend highly on the learning
	rate (such dependencies are not included in this simulation).

	Args:
		n_steps: An int specifying the number of training steps to simulate.

		bias: A float specifying the bias of the random walk used to simulate loss values.

		Optional keyword arguments specifying hyperparameter settings for the AdaptiveLearningRate object.

	Returns:
		None.
	'''

	learning_rate = AdaptiveLearningRate(**kwargs)
	save_step = n_steps/4

	loss = 0.
	loss_history = np.zeros([n_steps])
	rate_history = np.zeros([n_steps])
	for step in range(n_steps):
		if step == save_step:
			learning_rate.save('/tmp/alr_data.alr')

		loss = loss + bias + npr.randn()
		rate_history[step] = learning_rate.update(loss)
		loss_history[step] = loss

	# Test .restore
	restored_lr = AdaptiveLearningRate()
	restored_lr.restore('/tmp/alr_data.alr')
	restored_rate_history = np.zeros([n_steps])
	for step in range(save_step,n_steps):
		loss = loss_history[step]
		restored_rate_history[step] = restored_lr.update(loss)

	mean_abs_restore_error = np.mean(np.abs(rate_history[save_step:]-restored_rate_history[save_step:]))
	print('Avg abs diff between original and restored: %.3e' % mean_abs_restore_error)

	F = plt.figure(1)

	F.add_subplot(2,1,1)
	plt.plot(range(n_steps), rate_history)
	plt.plot(range(n_steps), restored_rate_history, linestyle='--')
	plt.ylabel('Learning rate')

	F.add_subplot(2,1,2)
	plt.plot(range(n_steps), loss_history)
	plt.ylabel('loss')
	plt.xlabel('step')

	plt.show()
