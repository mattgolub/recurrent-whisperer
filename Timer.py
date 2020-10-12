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

	Example usage, using prospective splits (default):

		# Build a timer object to profile three tasks.
		t = Timer(3)

		t.start()           	# Start the timer (required).

		run_task_1()        	# Run task 1 of 3.
		t1 = t.split('Task 1')	# Measure time taken for task 1.

		run_task_2()        	# Run task 2 of 3.
		t2 = t.split('Task 2')	# Measure time taken for task 2.

		run_task_3()        	# Run task 3 of 3.
		t3 = t.split('Task 3')	# Measure time taken for task 3.

		t.disp()            	# Print profile of timing.

		--> Total time: 16.00s
		-->     Task 1: 2.00s (12.5%)
		-->     Task 2: 8.00s (50.0%)
		-->     Task 3: 6.00s (37.5%)

	Example usage, using retrospective splits:

		# Build a timer object to profile three tasks.
		t = Timer(3, do_retrospective=True)

		t.start()           	# Start the timer (optional).

		t.split('Task 1')   	# Measure time taken for task 1.
		run_task_1()        	# Run task 1 of 3.

		t.split('Task 2')   	# Measure time taken for task 2.
		run_task_2()        	# Run task 2 of 3.

		t.split('Task 3')   	# Measure time taken for task 3.
		run_task_3()        	# Run task 3 of 3.

		t.stop()				# Stop the timer
								# (required to see split 3 time)

		t.disp()            	# Print profile of timing.

		--> Total time: 16.00s
		-->     Task 1: 2.00s (12.5%)
		-->     Task 2: 8.00s (50.0%)
		-->     Task 3: 6.00s (37.5%)

		# Note that split() returns None with retrospective splits, as it
		# cannot yet know the timing of the upcoming task.

	Outward facing usage is identical when the number of tasks to be timed is
	not specified from the outset. However, this prevents Timer from
	preallocating all required memory ahead of start(). Thus, Timer needs to
	allocate (very small) amounts of memory during timing, which may bias total
	time in unpredictable ways when total time is fast relative to memory
	allocations.
	'''

	def __init__(self,
		n_splits=10,
		do_retrospective=False,
		do_print_single_line=False,
		n_indent=0,
		name='Total',
		verbose=False):
		'''Builds a timer object.

		Args:
			n_splits (optional): int specifying the total number of splits to
			be timed. If provided, memory can be preallocated, which may
			prevent unpredictable allocation artifacts during profiling.
			Over-allocating (i.e., requesting more splits than ultimately are
			invoked) will not impact profiling (though may have memory
			implications in extreme applications). Default: 10.

			n_indent (optional): int specifying the number of indentation
			to prefix into print statements. Useful when utilizing multiple
			timer objects for profile nested code. Default: 0.

			name (optional): string specifying name for this timer, used only
			when printing updates. Default: 'Total'.

			verbose (optional): Bool indicating whether to print when
			allocating new splits beyond the initial n_splits (which may
			indicate biased timing for very fast splits). Default: False.

		Returns:
			None.
		'''

		assert n_splits >= 0, ('n_splits must be >= 0, but was %d' % n_splits)

		assert n_indent >= 0,('n_indent must be >= 0, but was %d' % n_indent)

		self.name = name
		self.do_retrospective = do_retrospective
		self.verbose = verbose

		'''Pre-allocate to avoid having to call append after starting the timer
		(which might incur non-uniform overhead, biasing timing splits).
		If more times are recorded than pre-allocated, lists will append.

		Note that self.times has n+1 elements (to include a start value) while
		self.task_names has n elements.
		'''
		self._empty_split_val = -1.0

		# Preallocate memory
		self._split_starts = [self._empty_split_val for idx in range(n_splits)]
		self._split_stops = [self._empty_split_val for idx in range(n_splits)]
		self._split_names = ['Task %d' % (idx+1) for idx in range(n_splits)]

		self._print_prefix = self._generate_print_prefix(n_indent)
		self._do_print_single_line = do_print_single_line

		self._is_started = False
		self._is_stopped = False

		''' Strategy for memory management:

		n: always represents the current length of
			_split_starts
			_split_stops
			_task_names
		'''
		self._alloc_len = n_splits
		self._idx = -1

	def __call__(self):
		'''Returns the time elapsed since the timer was started.
		   If start() has not yet been called, returns 0.
		'''

		if self._is_stopped:
			return self._stop_time - self._start_time
		if self._is_started:
			return time.time() - self._start_time
		else:
			return 0.0

	def start(self):
		'''Starts the timer. '''

		assert not self._is_stopped, 'Cannot restart a stopped Timer.'

		if self._is_started:
			self._print('Timer has already been started. '
				'Ignoring call to Timer.start()')
		else:
			self._is_started = True
			self._start_time = time.time()
			assert self._idx == -1, 'Inconsistent Timer state.'

			if self.do_retrospective:
				self._start_split()

	def stop(self):
		''' Stops the timer, freezing total and split times. '''

		# Future work: fill in final split if it had been started?

		if self._is_stopped:
			self._print('Timer has already been stopped. '
				'Ignoring call to Timer.stop()')
		else:
			self._stop_time = time.time()

	def split(self, name=None, stop=False):
		'''Records and returns the time elapsed for the most recent task.

		Args:
			name (optional): A string describing the most recent task.

		Returns:
			float indicating the split time in seconds.
		'''

		assert not self._is_stopped, \
			'Cannot take a split on a stopped Timer.'

		idx = self._idx # get idx before it's incremented

		if self.do_retrospective:
			'''
			Record split stop.
			Prepare and start next split.
			Return split time.
			'''

			assert self._is_running,\
				'Cannot record split time because Timer was not started.'

			assert self._split_starts[idx] != self._empty_split_val, \
				('Attempting to record split stop with no split start.')

			self._stop_split(name)

			if stop:
				# Avoid allocating a new split if known to be unnecessary.
				self.stop()
			else:
				self._start_split()

			return self._get_split_time(idx)

		else:

			if self._is_started:
				self._stop_split()
			else:
				self.start()

			self._start_split(name)

			return None

	def get_split(self, name):
		''' Retrieves a previously recorded split time.

		Args:
			name: the string name used to record the split, as previously
			provided in the call: split(name).

		Returns:
			float indicating the split time in seconds.
		'''

		idx = self.split_names.index(name)
		return self._get_split_time(idx)

	def _start_split(self, name=None):

		idx = self._prepare_next_split()

		if not self.do_retrospective and name is not None:
			self._split_names[idx] = name

		self._split_starts[idx] = time.time()

	def _stop_split(self, name=None):

		idx = self._idx

		self._split_stops[idx] = time.time()

		if self.do_retrospective and name is not None:
			self._split_names[idx] = name

	def disp(self, *args, **kwargs):

		Print('Timer.disp() is deprecated and '
		      'will be removed in a future version of Timer.py. '
		      'Use Timer.print(...) instead.')

		self.print(*args, **kwargs)

	def print(self, n_indent=None):
		'''Prints the profile of the tasks that have been timed thus far.

		Args:
			None.

		Returns:
			None.
		'''

		if self._is_started:
			total_time = self._print_total_time(n_indent=n_indent)
			self._print_split_times(total_time, n_indent=n_indent)
		else:
			self._print('Timer has not been started.')

	@property
	def _is_running(self):
		'''Returns a bool indicating whether or not the timer has been started.
		'''
		return self._is_started and not self._is_stopped

	def _is_split_complete(self, idx):
		''' Returns True if split[idx] has been completed, meaning it has a
		recorded start time and stop time.
		'''

		return idx < self._alloc_len and \
			self._split_starts[idx] != self._empty_split_val and \
			self._split_stops[idx] != self._empty_split_val

	def _get_split_time(self, idx):

		assert self._is_split_complete(idx), \
			('split[%d] is not complete.' % idx)

		return self._split_stops[idx] - self._split_starts[idx]

	def _prepare_next_split(self):
		# This is the only place that _idx and _alloc_len are ever changed.

		assert self._idx == -1 or self._is_split_complete(self._idx),\
			('Cannot prepare split %d because split %d is not complete.' %
			(self._idx+1, self._idx))

		self._idx += 1

		# Ensure safe to write to _split_times[idx], etc.
		if self._idx == self._alloc_len:

			if self.verbose:
				self._print('Appending Timer lists. '
					'This may cause biased time profiling.')

			self._split_starts.append(self._empty_split_val)
			self._split_stops.append(self._empty_split_val)
			self._split_names.append('Task %d' % (self._idx+1))
			self._alloc_len += 1

		return self._idx

	def _print_split_times(self, total_time,
		n_indent=None,
		do_single_line=None):
		# Print split times for all completed splits

		if n_indent is None:
			prefix = self._print_prefix
		else:
			prefix = self._generate_print_prefix(n_indent)

		if do_single_line is None:
			do_single_line = self._do_print_single_line

		if do_single_line:
			prefix = ' '
			end = ''
			print('[', end=end)
		else:
			end = '\n'
			prefix = prefix + '\t'

		idx = 0
		while self._is_split_complete(idx):

			split_name = self._split_names[idx]
			split_time = self._get_split_time(idx)

			print('%s%s: %.2fs (%.1f%%)' %
				(prefix,
				split_name,
				split_time,
				100.*split_time/total_time),
				end=end)

			idx += 1

		if self._do_print_single_line:
			print(' ]', end='\n')

	def _print_total_time(self, n_indent=None, do_single_line=None):
		# Print total time

		if n_indent is None:
			prefix = self._print_prefix
		else:
			prefix = self._generate_print_prefix(n_indent)

		if do_single_line is None:
			do_single_line = self._do_print_single_line

		total_time = self.__call__()
		print_data = (prefix, self.name, total_time)
		end = '' if do_single_line else '\n'
		print('%s%s time: %.2fs: ' % print_data, end=end)

		return total_time

	def _print(self, str, n_indent=None):
		'''Prints string after prefixing with the desired number of
		indentations.

		Args:
			str: The string to be printed.

		Returns:
			None.
		'''

		if n_indent is None:
			prefix = self._print_prefix
		else:
			prefix = self._generate_print_prefix(n_indent)

		print('%s%s' % (print_prefix, str))

	def _generate_print_prefix(self, n_indent):

		return '\t' * n_indent