'''
test_Hyperparameters.py
Test suite for some of Hyperparameters' key features.

Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.insert(0, '../')
from Hyperparameters import Hyperparameters

def test1():
    defaults = {
        'a': 'a1',
        'b': {'c': 'c1'},
        'd': 'd1',
    }

    non_hash = {
        'e': {
            'f': 'f1',
            'g': 'g1'
        }
    }

    inputs = {
        'a': 'a2', # This overrides a hash default
        'b': {'c': 'c2'}, # This overrides a hash default
        'e': {'g': 'g2'} # This overrides a non-hash default
    }

    correct_integration = {
        'a': 'a2',
        'b': {'c': 'c2'},
        'd': 'd1',
        'e': {
            'f': 'f1',
            'g': 'g2',
        }
    }

    correct_hash = {'a': 'a2', 'b': {'c': 'c2'}}

    # Test integration of hps
    hps = Hyperparameters(inputs, defaults, non_hash)
    print_test_result(hps._integrated_hps,
                correct_integration,
                'test 1a: simple nested hp integration with defaults')

    # Test partitioning for hashing
    print_test_result(hps.hps_to_hash,
                correct_hash,
                'test 1b: hp partitioning for hashing')

    # Test base case __getattr__ (no recursion necessary)
    print_test_result(hps['a'],
                'a2',
                'test 1c: base __getattr__ (no recursion)')

    # Test recursive __getattr__
    print_test_result(hps['b:c'],
                'c2',
                'test 1d: recursive __getattr__')

    # Test recursive __setattr__
    hps['e:g'] = 'g2'
    print_test_result(hps['e:g'],
                'g2',
                'test 1e: recursive __setattr__')

    # Test recursive __setattr__ for originally non-existing item
    # (hopefully this never happens in practice)
    hps['e:h'] = 'h2'
    print_test_result(hps['e:h'],
                'h2',
                'test 1f: recursive __setattr__, non-existing item')

def test2():
    # More complex integration test
    defaults = {
        'a': 'a1',
        'b': {'c': 'c1'},
        'd': 'd1',
        'e': {
            'f': 'f1',
            'g': 'g1',
            'h': 'h1',
            }
        }
    non_hash = {
        'i': 'i1',
        'j': {
            'k': 'k1',
            'l': 'l1',
        }
    }

    inputs = {
        'd': 'd2', # Simple override
        'e': {'g': 'g2'}, # Complex hash override, want to only override 'g',
                          # but without affecting 'f' or 'h'.
        'j': {'l': 'l2'}  # Complex non-hash override, want to only override
                          # 'l', but without affecting 'k'
    }

    correct_integration = {
        'a': 'a1',
        'b': {'c': 'c1'},
        'd': 'd2',
        'e': {
            'f': 'f1',
            'g': 'g2',
            'h': 'h1',
            },
        'i': 'i1',
        'j': {
            'k': 'k1',
            'l': 'l2',
            }
        }

    hps = Hyperparameters(inputs, defaults, non_hash)

    print_test_result(hps._integrated_hps,
                correct_integration,
                'test 2a: complex hp integration, partial dict overrides.')

    # These will be wrong or throw access errors if implementation is wrong
    print_test_result(hps['e:g'],
                'g2',
                'test 2b: recursive __getattr__') # should be 'g2'
    print_test_result(hps['j:k'],
                'k1',
                'test 2c: recursive __getattr__') # should be 'k1'

    # Test partitioning for hashing
    # should include d and g but NOT l
    print_test_result(hps.hps_to_hash,
                {'d': 'd2', 'e': {'g': 'g2'}},
                'test 2d: hp partitioning for hashing')

def test3():
    # Test integrate_hps using the sub/super application
    # See docstring for integrate_hps
    super_def_hps = {'a': 1, 'b': 2, 'c_hps': {'d': 3, 'e': 4}}
    sub_def_hps = {'b': 5, 'c_hps': {'d': 6}}
    correct_result = {'a': 1, 'b': 5, 'c_hps': {'d': 6, 'e': 4}}
    result = Hyperparameters.integrate_hps(super_def_hps, sub_def_hps)

    print_test_result(result,
                correct_result,
                'test 3a: subclass/superclass hp integration')

    print_test_result(super_def_hps,
                {'a': 1, 'b': 2, 'c_hps': {'d': 3, 'e': 4}},
                'test 3b: integration effects on original superclass hps')

    print_test_result(sub_def_hps,
                {'b': 5, 'c_hps': {'d': 6}},
                'test 3c: integration effects on original subclass hps')

def test4():
    # Same as test1, but also tests wildcard functionality
    defaults = {
        'a': 'a1',
        'b': {'c': 'c1'},
        'd': 'd1'
    }

    non_hash = {
        'e': {
            'f': 'f1',
            'g': 'g1'
        }
    }

    inputs = {
        '_wild1': 'w11',    # WILDCARD (no default, forced to hash)
        'a': 'a2',          # This overrides a hash default
        'b': {'c': 'c2'},   # This overrides a hash default
        'e': {
            'g': 'g2',      # This overrides a non-hash default
            '_wild2': 'w21' # WILDCARD (no default, forced to hash)
        },
    }

    correct_hash = {
        '_wild1': 'w11',
        'a': 'a2',
        'b': {'c': 'c2'},
        'e': {
            '_wild2': 'w21'
        },
    }

    correct_integration = {
        '_wild1': 'w11',
        'a': 'a2',
        'b': {'c': 'c2'},
        'd': 'd1',
        'e': {
            'f': 'f1',
            'g': 'g2',
            '_wild2': 'w21'
        },
    }

    # Test integration of hps
    hps = Hyperparameters(inputs, defaults, non_hash)

    print_test_result(hps._integrated_hps,
                correct_integration,
                'test 4a: integration with wildcard (_)')

    print_test_result(hps.hps_to_hash,
            correct_hash,
            'test 4b: ensure wildcards are hashed')

def print_test_result(result, correct, name, verbose=False):

    if result == correct:
        print('Passed: %s.' % name)
    else:
        print('Failed %s.' % name)
        verbose = True

    if verbose:

        if isinstance(correct, str):
            correct_str = correct
            result_str = result
        else:
            correct_str = Hyperparameters.printable_str_from_dict(correct)
            result_str = Hyperparameters.printable_str_from_dict(result)

        print('\nCorrect result:\n%s' % correct_str)
        print('\nHyperparameters result:\n%s' % result_str)

def main():

    test1()
    print()

    test2()
    print()

    test3()
    print()

    test4()
    print()

if __name__ == '__main__':
    main()