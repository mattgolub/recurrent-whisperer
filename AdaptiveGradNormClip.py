'''
AdaptiveGradNormClip.py
Version 1.0
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

import numpy as np
import cPickle

class AdaptiveGradNormClip(object):
	"""Class for managing adaptive gradient norm clipping for stabilizing any gradient-descent-like procedure.

	Essentially, just a running buffer of gradient norms from the last n gradient steps, with a hook into the x-th percentile of those values, which is intended to be used to set the ceiling on the gradient applied at the next iteration of a gradient-descent-like procedure.

	The standard usage is as follows:

	```python
	# Set hyperparameters as desired.
	agnc_hps = dict()
	agnc_hps['sliding_window_len'] = 1.0
	agnc_hps['percentile'] = 95
	agnc_hps['init_clip_val' = 1.0
	agnc_hps['verbose'] = False
	agnc = AdaptiveGradNormClip(**agnc_hps)

	while some_conditions(...):
		# This loop defines one step of the training procedure.

		gradients = get_gradients(data, params)
		grad_norm = compute_gradient_norm(gradients)
		clip_val = agnc.update(grad_norm)
		clipped_gradients = clip_gradients(gradients, clip_val)
		params = apply_gradients(clipped_gradients)

		# (Optional): Occasionally save model checkpoints along with the
		# AdaptiveGradNormClip object (for seamless restoration of a training
		# session)
		if some_other_conditions(...):
			save_checkpoint(params, ...)
			agnc.save(...)
	```

	"""
	def __init__(self,
		sliding_window_len=128,
		percentile=95,
		init_clip_val=1e12,
		verbose=False):
		'''Builds an AdaptiveGradNormClip object

		Args:
			A set of optional keyword arguments for overriding the default
			values of the following hyperparameters:

			sliding_window_len: An int specifying the number of recent steps to
			record. Default: 100.

			percentile: A float between 0.0 and 100.0 specifying the percentile
			of the recorded gradient norms at which to set the clip value.
			Default: 95.

			init_clip_val: A float specifying the initial clip value (i.e., for
			step 1, before any empirical gradient norms have been recorded).
			Default: 1e12.

				This default effectively prevents any clipping on iteration one.
				This has the unfortunate side effect of throwing the vertical
				axis scale on the corresponding Tensorboard plot. The
				alternatives are computationally inefficient: either clip at an
				arbitrary level (or at 0) for the first epoch or compute a
				gradient at step 0 and initialize to the norm of the global
				gradient.

			verbose: A bool indicating whether or not to print status updates.
			Default: False.
		'''
		self.step = 0
		self.sliding_window_len = sliding_window_len
		self.percentile = percentile
		self.clip_val = init_clip_val
		self.grad_norm_log = []
		self.verbose = verbose

	def __call__(self):
		'''Returns the current clip value.

		Args:
			None.

		Returns:
			A float specifying the current clip value.
		'''
		return self.clip_val

	def update(self, grad_norm):
		'''Update the log of recent gradient norms and the corresponding
		recommended clip value.

		Args:
			grad_norm: A float specifying the gradient norm from the most
			recent gradient step.

		Returns:
			None.
		'''
		if self.step < self.sliding_window_len:
			# First fill up an entire "window" of values
			self.grad_norm_log.append(grad_norm)
		else:
			# Once the window is full, overwrite the oldest value
			idx = np.mod(self.step, self.sliding_window_len)
			self.grad_norm_log[idx] = grad_norm

		self.clip_val = np.percentile(self.grad_norm_log, self.percentile)

		self.step += 1

	def save(self, save_path):
		'''Saves the current AdaptiveGradNormClip state, enabling seamless restoration of gradient descent training procedure.

		Args:
			save_path: A string containing the path for saving the current object state.

		Returns:
			None.
		'''

		if self.verbose:
			print('Saving AdaptiveGradNormClip.')
		file = open(save_path,'w')
		file.write(cPickle.dumps(self.__dict__))
		file.close

	def restore(self, restore_path):
		'''Loads a previously saved AdaptiveGradNormClip state, enabling seamless restoration of gradient descent training procedure.

		Args:
			restore_path: A string containing the path from which to load the object state.

		Returns:
			None.
		'''
		if self.verbose:
			print('Restoring AdaptiveGradNormClip.')
		file = open(restore_path,'r')
		restore_data = file.read()
		file.close()
		self.__dict__ = cPickle.loads(restore_data)
