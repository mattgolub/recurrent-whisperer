'''
Timer.py
Version 1.0
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

import numpy as np
import time

class Timer(object):
	'''Class for profiling computation time.

	Example usage:
		t = Timer(3)		# Build a timer object to profile three tasks.
		t.start()    		# Start the timer.

		run_task_1() 		# Run task 1 of 3.
		t.split('Task 1')	# Measure time taken for task 1.

		run_task_2() 		# Run task 2 of 3.
		t.split('Task 2')	# Measure time taken for task 2.

		run_task_3() 		# Run task 3 of 3.
		t.split('Task 3')	# Measure time taken for task 3.

		t.disp()			# Print profile of timing.

		--> Total time: 16.00s
		--> 	Task 1: 2.00s (12.5%)
		--> 	Task 2: 8.00s (50.0%)
		--> 	Task 3: 6.00s (37.5%)
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

		self.n = n_tasks
		self.t = np.zeros(n_tasks+1)

		# Pre-allocate to avoid having to call append after starting the timer
		# (which might incur non-uniform overhead, biasing timing splits)
		self.task_names = []
		for idx in range(n_tasks):
			self.task_names.append(None)

		''' Note that self.t has n+1 elements (to include a start value) while
		self.task_names has n elements'''

		self.print_prefix = '\t' * n_indent
		self.name = name

		self.idx = 0
		self.is_running = False

	def __call__(self):
		'''Returns the time elapsed since the timer was started.
		   If start() has not yet been called, returns NaN.
		'''
		if self.is_running:
			return time.time() - self.t[0]
		else:
			return 0.0

	def start(self):
		'''Starts the timer'''

		self.is_running = True
		self.idx = 1
		self.t[0] = time.time()

	def split(self, task_name=None):
		'''Measures the time elapsed for the most recent task.

		Args:
			task_name (optional): A string describing the most recent task.

		Returns:
			None.
		'''

		if self.is_running:
			self.t[self.idx] = time.time()
			self.task_names[self.idx - 1] = task_name
			self.idx += 1
		else:
			print('%sTimer cannot take a split until it has been started.' %
			      self.print_prefix)

	def disp(self):
		'''Prints the profile of the tasks that have been timed thus far.

		Args:
			None.

		Returns:
			None.
		'''
		if self.idx > 1:
			total_time = self.t[self.idx-1] - self.t[0]
			split_times = np.diff(self.t)

			print('%s%s time: %.2fs'
				% (self.print_prefix, self.name, total_time))

			# allows printing before all tasks have been run
			for idx in range(self.idx-1):
				if self.task_names[idx] is None:
					task_name = 'Task' + str(idx+1)
				else:
					task_name = self.task_names[idx]
				print('\t%s%s: %.2fs (%.1f%%)'
					% (self.print_prefix,
					   task_name,
					   split_times[idx],
					   100*split_times[idx]/total_time))
		else:
			print('%sTimer has not yet taken any splits to time.' %
			      self.print_prefix)