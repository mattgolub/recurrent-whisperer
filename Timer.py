'''
Timer.py
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

class Timer(object):
	'''Class for profiling computation time.

	Example usage:
		t = Timer(3)        # Build a timer object to profile three tasks.
		t.start()           # Start the timer.

		run_task_1()        # Run task 1 of 3.
		t.split('Task 1')   # Measure time taken for task 1.

		run_task_2()        # Run task 2 of 3.
		t.split('Task 2')   # Measure time taken for task 2.

		run_task_3()        # Run task 3 of 3.
		t.split('Task 3')   # Measure time taken for task 3.

		t.disp()            # Print profile of timing.

		--> Total time: 16.00s
		-->     Task 1: 2.00s (12.5%)
		-->     Task 2: 8.00s (50.0%)
		-->     Task 3: 6.00s (37.5%)
	'''

	def __init__(self, n_tasks=1, n_indent=0, name='Total'):
		'''Builds a timer object.

		Args:
			n_tasks: int specifying the total number of tasks to be timed.

			n_indent (optional): int specifying the number of indentation
				to prefix into print statements. Useful when utilizing multiple
				timer objects for profile nested code. Default: 0.

			name (optional): string specifying name for this timer, used only
				when printing updates. Default: 'Total'.

		Returns:
			None.
		'''

		if n_tasks < 0:
			raise ValueError('n_tasks must be >= 0, but was %d' % n_tasks)

		if n_indent < 0:
			raise ValueError('n_indent must be >= 0, but was %d' % n_indent)

		self.n = n_tasks

		'''Pre-allocate to avoid having to call append after starting the timer
		(which might incur non-uniform overhead, biasing timing splits).
		If more times are recorded than pre-allocated, lists will append.

		Note that self.times has n+1 elements (to include a start value) while self.task_names has n elements
		'''
		self.times = [np.nan for idx in range(n_tasks + 1)]
		self.task_names = [None for idx in range(n_tasks)]

		self.print_prefix = '\t' * n_indent
		self.name = name

		self.idx = -1

	def __call__(self):
		'''Returns the time elapsed since the timer was started.
		   If start() has not yet been called, returns NaN.
		'''
		if self.is_running():
			return time.time() - self.times[0]
		else:
			return 0.0

	def start(self):
		'''Starts the timer'''

		if self.is_running():
			self._print('Timer has already been started. '
				'Ignoring call to Timer.start()')
		else:
			self.idx += 1
			self.times[self.idx] = time.time()
	def is_running(self):
		'''Returns a bool indicating whether or not the timer has been started.
		'''
		return self.idx >= 0

	def split(self, task_name=None):
		'''Measures the time elapsed for the most recent task.

		Args:
			task_name (optional): A string describing the most recent task.

		Returns:
			None.
		'''

		if self.is_running():
			self.idx += 1
			if self.idx <= self.n:
				self.times[self.idx] = time.time()
				self.task_names[self.idx - 1] = task_name
			else:
				self.times.append(time.time())
				self.task_names.append(task_name)
				self.n += 1
				self._print('Appending Timer lists. '
					'This may cause biased time profiling.')
		else:
			self._print('Timer cannot take a split until it has been started.')

	def disp(self, print_on_single_line=False):
		'''Prints the profile of the tasks that have been timed thus far.

		Args:
			None.

		Returns:
			None.
		'''

		if not self.is_running():
			self._print('Timer has not been started.')
		elif self.idx == 0:
			self._print('Timer has not yet taken any splits to time.')
		else:
			total_time = self.times[self.idx] - self.times[0]
			split_times = np.diff(self.times)

			print_data = (self.print_prefix, self.name, total_time)
			if print_on_single_line:
				print('%s%s time: %.2fs: ' % print_data, end='')
			else:
				print('%s%s time: %.2fs' % print_data)

			# allows printing before all tasks have been run
			for idx in range(self.idx):
				if self.task_names[idx] is None:
					task_name = 'Task ' + str(idx+1)
				else:
					task_name = self.task_names[idx]

				if print_on_single_line:
					print(' %s: %.2fs (%.1f%%); ' %
						(task_name,
						split_times[idx],
						100*split_times[idx]/total_time),
						end='')
				else:
					print('\t%s%s: %.2fs (%.1f%%)' %
						(self.print_prefix,
					   	task_name,
					   	split_times[idx],
					   	100*split_times[idx]/total_time))

			if print_on_single_line:
				print('')

	def _print(self, str):
		'''Prints string after prefixing with the desired number of indentations.

		Args:
			str: The string to be printed.

		Returns:
			None.
		'''

		print('%s%s' % (self.print_prefix, str))
