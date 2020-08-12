'''
Hyperparameters.py
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pdb
from copy import copy, deepcopy
import hashlib
import cPickle
from numpy import sort
import yaml

class Hyperparameters(object):
    '''A general class for managing a model's hyperparameters, default
    settings, and hashes for organizing model checkpoint directories.

    '''

    _delimiter = ':'

    def __init__(self,
                 hps=dict(),
                 default_hash_hps=dict(),
                 default_non_hash_hps=dict(),
                 hash_len=10,
                 verbose=False):
        '''Creates a Hyperparameters object.

        The resulting object contains hyperparameters as class attributes,
        with default values overridden by the settings in hps.

        Args:
            hps: dict containing hyperparameter names as keys and
            corresponding settings as values. These settings are used to
            override default values. All values must be also included in either
            default_hash_hps or default_non_hash_hps. Default: None.

            default_hash_hps: dict containing hyperparameters and their
            default values for the set of hyperparameters to be hashed. These
            hyperparameters may affect the model architecture or the
            trajectory of fitting, and as such are typically swept during
            hyperparameter optimization.

            default_non_hash_hps: dict containing hyperparameters and their
            default values for the set of hyperparameters that are not
            included in the hash. These hyperparameters should not influence
            the model architecture or the trajectory of fitting.

            hash_len (optional): int between 1 and 512 specifying the number
            of hex characters to include in the hyperparameters hash (all
            others are truncated). Larger values in this range may be
            necessary for massive hyperparameter searches, where the
            likelihood of hash collisions may be non-negligible. Default: 10.

            verbose (optional): bool indicating whether to print status
            updates. Default: False.

        Returns:
            None.

        Raises:
            ValueError if at least one default_hash_hps or
            default_non_hash_hps is not specified.
        '''

        self._validate_args(hps, default_hash_hps, default_non_hash_hps)

        self.default_hash_hps = default_hash_hps
        self.default_non_hash_hps = default_non_hash_hps

        self.default_hps = {}
        self.default_hps.update(default_hash_hps)
        self.default_hps.update(default_non_hash_hps)

        # dict containing all hyperparameters set to defaults, except as
        # overridden by values in hps.
        self.integrated_hps = self.integrate_hps(self.default_hps, hps)

        self._hash_len = hash_len
        self._verbose = verbose
        for key, val in self.integrated_hps.iteritems():
            setattr(self, key, val)

    # ************************************************************************
    # Exposed functions ******************************************************
    # ************************************************************************

    @staticmethod
    def parse_command_line(default_hps, description=None):
        '''Parses command-line arguments into a dict of hyperparameters.

        Args:
            default_hps: dict specifying all default hyperparameter settings.
            All values must have type bool, int, float, str, or dict. Values
            that are dicts must themselves have values of only the
            aforementioned types.

                Nesting of dicts is supported at the command line using
                colon delimiters, e.g.:

                    python your_script.py --d1:d2:n 10

                will update the value in default_hps['d1']['d2']['n'] to be 10.

            description (optional): string containing text to display if help
            commands (-h, --help) are invoked at the command line. See
            argparse.ArugmentParser for further details. Default: None.

        Returns:
            dict matching default_hps, but with values replaced by any
            corresponding values provided at the command line.
        '''

        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        def parse_helper(D, str_prefix=''):
            '''
            Validates value types and recurses appropriately when encountering
            colon delimiters in keys.

            Args:
                D: dict containing default hyperparameter values. Values must
                be of type bool, int, float, str, or dict.

            Returns:
                None.
            '''
            for key, val in D.iteritems():

                if val is None or val=='None':
                    # Don't set default. This results in default set to None.
                    # setting default=None can result in default='None' (bad).
                    parser.add_argument('--' + str_prefix + str(key))

                elif isinstance(val, bool) or \
                    isinstance(val, int) or \
                    isinstance(val, float) or \
                    isinstance(val, str):

                    if isinstance(val, bool):
                        type_val = str2bool
                    elif val is not None:
                        type_val = type(val)

                    parser.add_argument('--' + str_prefix + str(key),
                        default=val, type=type_val)

                elif isinstance(val, dict):
                    # Recursion
                    parse_helper(val, str_prefix=str_prefix + str(key) + ':')

                else:
                    raise argparse.ArgumentTypeError('Default value must be '
                        'bool, int, float, str, dict, or None, but was %s. ' %
                        type(val))

        parser = argparse.ArgumentParser(description=description)
        parse_helper(default_hps)

        args = parser.parse_args()

        # This has no dicts (all values are bool, int, float, or str)
        hps_flat = vars(args)

        # Recursively reconstruct any dicts (based on colon delimiters)
        hps = Hyperparameters.unflatten(hps_flat)

        return hps

    @staticmethod
    def integrate_hps(hps, update_hps):
        '''Integrates two hyperparameter dicts, with update_hps overriding and
        possibly adding novel items to hps.

        Integration is done recursively to support selective updating within
        hyperparameters that are themselves hyperparameters dicts (e.g.,
        containing hyperparameters for helper classes).

        Note: This function does not perform any checks on the structure of
        hps or update_hps. See example application 1 below. If needed, such
        checks should be performed before calling this function.

        Example:

            super_def_hps = {'a': 1, 'b': 2, 'c_hps': {'d': 3, 'e': 4}}
            sub_def_hps = {'b': 5, 'c_hps': {'d': 6}}
            integrate_hps(super_def_hps, sub_def_hps)
                --> {'a': 1, 'b': 5, 'c_hps': {'d': 6, 'e': 4}}

            Notice that c_hps does not become sub_def_hps['c_hps'], but rather
            only c_hps['d'] is changed from super_def_hps['c_hps'].

        Example application:

        1.  Overriding default hps.

                integrated_hps = integrate_hps(default_hps, input_hps)

            This is the use case found in Hyperparameters.__init__(). Here,
            default_hps contains the set of all possible hyperparameters along
            with their default settings. Those defaults are then updated by
            input_hps, which in this use case would contain a subset of the
            hyperparameters in default_hps, along with values for overriding
            those default settings. In this case, it might be necessary to
            ensure that all input_hps indeed have a corresponding entry in
            default_hps (e.g., to protect against user error). This check is
            not performed by integrate_hps--for Hyperparameters, that check is
            in _validate_args()).

        2.  Integrating a superclass' default hyperparameters with the
            hyperparameters of a subclass:

                integrated_hps = integrate_hps(super_hps, subclass_hps)

            Subclass hyperparameters will override defaults specified by this
            superclass, and possibly add new hyperparameters not present in
            the superclass.

        Args:
            hps: dict containing hyperparameters to be updated.

            update_hps: dict containing updates (overrides + additions) to hps.

        Returns:
            integrated_hps: dict containing the integrated hyperparameters.
        '''

        # copy needed to prevent infinite recursive links between data
        # structures in recursive models. deepcopy may be overkill.
        integrated_hps = deepcopy(hps)
        update_hps = deepcopy(update_hps)

        for key, val in update_hps.iteritems():

            if not isinstance(val, dict) or key not in integrated_hps:
                # Base case
                integrated_hps[key] = val
            else:
                # Recurse
                integrated_hps[key] = Hyperparameters.integrate_hps(
                    integrated_hps[key], val)

        return integrated_hps

    @property
    def hps_to_hash(self):
        return self._get_hash_hps(self.integrated_hps, self.default_hash_hps)

    @property
    def hash(self):
        '''Computes a hash of all non-default hash hyperparameters.
        In most applications (e.g., setting up run directories), this is the
        hash that should be used. By omitting default hps, hashes become more
        robust to future updates to an algorithm--if new hps are introduced,
        and their default values are chosen to reproduce before-update
        behavior, before-update and after-update hashes will be compatible.

        Args:
            None.

        Returns:
            string containing the hyperparameters hash.
        '''
        return self._generate_hash(self.hps_to_hash)

    @property
    def _hash_all_hps(self):
        '''Computes a hash of all hyperparameters, including default and
        non-hash hyperparameters. This function is provided as a convenience
        for testing.

        Args:
            None.

        Returns:
            string containing the hyperparameters hash.
        '''
        return self._generate_hash(self.integrated_hps)

    def save(self, save_path):
        '''Saves the Hyperparameters object.

        Args:
            save_path: string containing the path at which to save these
            hyperparameter settings (including filename and arbitrary
            extension).

        Returns:
            None.
        '''
        self._maybe_print('Saving Hyperparameters.')
        file = open(save_path, 'wb')
        file.write(cPickle.dumps(self.integrated_hps))
        file.close()

    def save_yaml(self, save_path):
        # Note, this doesn't do well with numpy variables.
        # Ideally, make sure everything is bool, int, float, string, etc.
        self._maybe_print('Writing Hyperparameters YAML file.')
        with open(save_path, 'w') as yaml_file:
            yaml.dump(self.integrated_hps,
                      yaml_file,
                      default_flow_style=False)

    @staticmethod
    def restore(restore_path):
        '''Loads hyperparameters from a saved Hyperparameters object.

        Args:
            restore_path:
                string containing the path of a previously saved
                Hyperparameters object (including filename and extension).

        Returns:
            dict containing all hyperparameter names and settings from the
            previously saved Hyperparameters object.
        '''
        file = open(restore_path, 'rb')
        restore_data = file.read()
        file.close()
        return cPickle.loads(restore_data)

    def restore_from_yaml(self, yaml_path):
        # Note, this returns a dict of hps, not a Hyperparameters object
        # This is for use on runs created before save/restore was implemented
        # Preferred usage is with save/restore

        self._maybe_print('Reading Hyperparameters YAML file.')
        with open(yaml_path, 'r') as yaml_file:
            return yaml.load(yaml_file)

    @staticmethod
    def flatten(D, delimiter=_delimiter):
        ''' Flattens a dict: Values that are themselves dicts are recursively
        'flattened' by concatenating keys using colon delimiting.

        Example:
            D = {'a': 1, 'b': {'c':2, 'd':3}}
            flatten(D)
            {'a': 1, 'b:c': 2, 'b:d': 3}

        Args:
            D: Python dict.

        Returns:
            D_flat: The flattened dict.
        '''

        D_flat = dict()
        for key, val in D.iteritems():

            assert (isinstance(key, str)),\
                ('Keys must be strings, '
                 'but found one of type: %s' % str(type(key)))

            if isinstance(val, dict):
                val_flat = Hyperparameters.flatten(val)
                for key2, val2 in val_flat.iteritems():
                    D_flat[key + delimiter + key2] = val2
            else:
                D_flat[key] = val

        return D_flat

    @staticmethod
    def unflatten(D_flat, delimiter=_delimiter):
        ''' Unflattens a flattened dict. A flattened dict is a dict with no
        values that are themselves dicts. Nested dicts can be represented in a
        flattened dict using colon-delimited string keys.

        Example:
            D_flat = {'a': 1, 'b:c': 2, 'b:d': 3}
            unflatten(D_flat)
            {'a': 1, 'b': {'c': 2, 'd': 3}}

        Args:
            D_flat: dict with values of type bool, int, float, or str (no
            dicts allowed).

        Returns:
            D_unflattened: dict, nested with other dicts as specified by
            colon delimiting in keys of D_flat.
        '''

        def assign_leaf(key, val):
            '''
            Handles the case where we are writing the first entry of a
            potentially nested dict. Because it's the first entry, there
            is no chance of overwriting an existing dict.
            '''

            assert (isinstance(key, str)),\
                ('Keys must be strings, '
                 'but found one of type: %s' % str(type(key)))

            if ':' in key:
                dict_name, rem_name = \
                    Hyperparameters._parse_delimited_hp_name(
                        key, delimiter=delimiter)
                return {dict_name: assign_leaf(rem_name, val)}
            else:
                return {key: val}

        def add_helper(D, key, val):

            assert (isinstance(key, str)),\
                ('Keys must be strings, '
                 'but found one of type: %s' % str(type(key)))

            if delimiter in key:

                dict_name, rem_name = \
                    Hyperparameters._parse_delimited_hp_name(
                        key, delimiter=delimiter)

                if dict_name in D:
                    D[dict_name] = add_helper(D[dict_name], rem_name, val)
                else:
                    D[dict_name] = assign_leaf(rem_name, val)

            else:
                D[key] = val

            return D

        D_unflattened = dict()
        for key, val in D_flat.iteritems():
            # Don't need to check type here (handled by parse_helper).
            D_unflattened = add_helper(D_unflattened, key, val)

        return D_unflattened

    # ************************************************************************
    # General class access and manipulation **********************************
    # ************************************************************************

    def __dict__(self):
        '''Returns hyperparameters in a dict.

        Args:
            None.

        Returns:
            dict of all hyperparameter names and settings as key, value pairs.
        '''
        return self.integrated_hps

    def __getitem__(self, key):
        '''Provides access to an individual hyperparameter value, with support
        for colon-delimited hyperparameter names.

        Args:
            key: A string indicating the name of the hyperparameter to be
            retrieved. Colon delimiting specifies traversals into
            (possibly nested) dicts within the master hyperparameters dict.

        Returns:
            The hyperparameter value corresponding to the name in key.
        '''

        return Hyperparameters._getitem(self.integrated_hps, key)

    @staticmethod
    def _getitem(D, key):
        # Helper function to recursively traverse into dict D.
        if ':' in key:

            dict_name, rem_name = \
                Hyperparameters._parse_colon_delimited_hp_name(key)

            return Hyperparameters._getitem(D[dict_name], rem_name)

        else:

            return D[key]

    def __setitem__(self, key, value):
        '''Assigns an individual hyperparameter value, with support for
        colon-delimited hyperparameter names.

        Args:
            key: A string indicating the name of the hyperparameter to be
            assigned. Colon delimiting specifies traversals into
            (possibly nested) dicts within the master hyperparameters dict.
            Nested dicts are created if they do not already exist.

            value: The value of the hyperparameter to be assigned.

        Returns:
            None.
        '''

        # Update the value in self.integrated_hps
        Hyperparameters._setitem(self.integrated_hps, key, value)

        ''' Update the value stored as a class attribute. This may or may not
        be necessary given how original attributes were setup.'''
        if ':' in key:
            # Replace the entire class attribute dict with the updated one
            dict_name, rem_name = self._parse_delimited_hp_name(key)
            setattr(self, dict_name, self.integrated_hps[dict_name])
        else:
            setattr(self, key, value)

    @staticmethod
    def _setitem(D, key, value):
        # Helper function to recursively traverse into dict D.
        if ':' in key:

            dict_name, rem_name = \
                Hyperparameters._parse_delimited_hp_name(key)

            if not dict_name in D.keys():

                D[dict_name] = dict()

            Hyperparameters._setitem(D[dict_name], rem_name, value)

        else:

            D[key] = value

    def __str__(self):

        def print_helper(D, n_indent=0):
            str = ''
            indent = n_indent * '\t'
            for key in sort(D.keys()):
                value = D[key]

                str += '%s%s: ' % (indent, key)
                if isinstance(value, dict):
                    str += '\n' + print_helper(value, n_indent+1)
                else:
                    str += '%s\n' % value.__str__()

            return str

        str = print_helper(self.integrated_hps)

        return str

    # ************************************************************************
    # Internal helper functions **********************************************
    # ************************************************************************

    @staticmethod
    def _validate_args(hps, default_hash_hps, default_non_hash_hps):
        ''' Checks to ensure the dicts provided to __init__ fit the
        requirements as stated in __init__().

        Args:
            See docstring for __init__.

        Returns:
            None.

        Raises:
            ValueError if any of the hps args are not dicts.

            ValueError if any key in hps is not included in exactly one of
            default_hash_hps or default_non_hash_hps.

            ValueError if there are any keys common to both default_hash_hps
            and default_non_hash_hps.
        '''

        Hyperparameters._validate_keys(hps)
        Hyperparameters._validate_keys(default_hash_hps)
        Hyperparameters._validate_keys(default_non_hash_hps)

        if not isinstance(hps, dict):
            raise ValueError('hps must be a dict but is not.')

        if not isinstance(default_hash_hps, dict):
            raise ValueError('default_hash_hps must be a dict but is not.')

        if not isinstance(default_non_hash_hps, dict):
            raise ValueError('default_non_hash_hps must be a dict but is not.')

        # Flatten all HP dicts to ensure validation recursively into HPs that
        # are themselves HP dicts..
        flat_hps = Hyperparameters.flatten(hps)
        flat_def_hash_hps = Hyperparameters.flatten(default_hash_hps)
        flat_def_non_hash_hps = Hyperparameters.flatten(default_non_hash_hps)

        input_set = set(flat_hps.keys())
        default_hash_set = set(flat_def_hash_hps.keys())
        default_non_hash_set = set(flat_def_non_hash_hps.keys())

        # Check to make sure each default hp is either hash or non-hash
        # (not both).
        default_intersection = \
            default_hash_set.intersection(default_non_hash_set)
        if len(default_intersection) > 0:
            for violating_key in default_intersection:
                print('Hyperparameter [%s] cannot be both hash '
                    'and non-hash.' % violating_key)
            raise ValueError('Overlapping default-hash and default-non-hash '
                             'hyperparameters.')

        # Check to make sure all input_hps are valid (i.e., have default
        # values) If not, user may have mistyped an hp name.
        default_union = default_hash_set.union(default_non_hash_set)
        if not input_set.issubset(default_union):
            error_msg = str('Attempted to override hyperparameter(s) '
                'with no defined defaults.\n')
            for violating_key in input_set - default_union:
                error_msg += str('\t%s is not a valid hyperparameter '
                    '(has no specified default).\n' % violating_key)
            raise ValueError(error_msg)

    @staticmethod
    def _validate_keys(hps):
        ''' Recursively checks that all keys are strings or dicts, and that
        string keys do not contain the delimiter.

        Args:
            hps: a (potentially nested) dict with keys as string
            hyperparameter names.

        Returns:
            None.

        Raises:
            ValueError if any keys are not strings or dicts.
        '''

        for key, val in hps.iteritems():

            assert (isinstance(key, str)),\
                ('Hyperparameters keys must be strings, '
                 'but found one of type: %s' % str(type(key)))

            assert Hyperparameters._delimiter not in key, \
                ('Hyperparameter keys cannot contain the delimiter (%s) '
                 'in the current implementation, but found \'%s\'' %
                 (Hyperparameters._delimiter, key))
            ''' If this becomes problematic, it might be worth reimplementing
            some of this class, where all HP dicts are immediately flattened
            and all internal manipulations are done on these flattened dicts.
            This would obviate all of the recursive functionality. Or a
            band-aid fix would be to flatten then unflatten all dicts in
            __init__().
            '''

            if isinstance(val, dict):
                Hyperparameters._validate_keys(val)

    @staticmethod
    def _get_hash_hps(hps, default_hash_hps):
        '''
        Returns:
            hps_to_hash: dict containing the subset of hyperparameters in
            input_hps that are to be hashed (i.e., keys with non-default
            values that are also keys in default_hash_hps).
        '''
        flat_hps = Hyperparameters.flatten(hps)
        flat_defaults = Hyperparameters.flatten(default_hash_hps)

        keys = flat_hps.keys()
        default_keys = flat_defaults.keys()

        ''' Because of the checks in _validate_args, we only need to check the
        intersection: Keys that are unique to hps are non-hash hps. Keys that
        are unique to default_hash_hps take default values and are not hashed.
        '''
        keys_to_check = set(keys).intersection(set(default_keys))

        # Add non-default entries to dict for hashing
        flat_hps_to_hash = dict()
        for key in keys_to_check:

            val = flat_hps[key]

            # Allows for 'None' to be used in command line arguments.
            if val == 'None':
                val = None

            # If value is not default, add it to the dict for hashing.
            if val != flat_defaults[key]:
                flat_hps_to_hash[key] = val

        hps_to_hash = Hyperparameters.unflatten(flat_hps_to_hash)

        return hps_to_hash

    def _generate_hash(self, hps):
        '''Generates a hash from a unique string representation of the
        hyperparameters in hps.

        Args:
            hps: dict of hyperparameter names and settings as keys and values.

        Returns:
            string containing 512-bit hash in hexadecimal representation.
        '''
        str_to_hash = self._sorted_str_from_dict(hps)

        # Generate the hash for that string
        h = hashlib.new('sha512')
        h.update(str_to_hash)
        hps_hash = h.hexdigest()[0:self._hash_len]

        return hps_hash

    @staticmethod
    def _sorted_str_from_dict(d):
        '''Creates a string of the key, value pairs in dict d, sorted by key.
        Sorting is required to 'uniquify' any set of hyperparameter settings
        (because two different orderings of the same hyperparameter settings
        are assumed to result in identical model and fitting behavior).

        Args:
            d: dict of hyperparameters.

        Returns:
            string of key, value pairs, sorted by key.
        '''

        sorted_keys = sort(d.keys())
        n_keys = len(sorted_keys)

        str_items = ['{']
        key_idx = 1
        for key in sorted_keys:
            val = d[key]

            if isinstance(val, dict):
                str_val = Hyperparameters._sorted_str_from_dict(val)
            else:
                str_val = str(val)

            new_entry = (str(key) + ': ' + str_val)
            str_items.append(new_entry)

            if key_idx < n_keys:
                str_items.append(', ')

            key_idx += 1

        str_items.append('}')

        return ''.join(str_items)

    @staticmethod
    def _parse_delimited_hp_name(hp_name, delimiter=':'):
        ''' Splits a string into the segments preceding and following the
        first colon. Used to indicate traversing into a sub-dict within a
        dict.

        Example: an hp_name with format "aa:bb:cc:..." is split into
        "aa" and "bb:cc:...". This is used elsewhere to indicate setting or
        getting D["aa"]["bb"]["cc"]...

        Args:
            hp_name is string containing at least 1 colon.

        Returns:
            dict_name: hp_name up to (but not including) the first colon.
            rem_name: hp_name from first char after colon to end.

        Raises:
            ValueError if hp_name does not contain any colons.
        '''
        if not ':' in hp_name:
            raise ValueError(
                'hp_name does not contain delimiter (%s).' % delimitter)

        first_idx = hp_name.index(delimiter)
        dict_name = hp_name[:first_idx]
        rem_name = hp_name[(first_idx + 1):]

        return dict_name, rem_name

    def _maybe_print(self, s):
        '''Depending on verbosity level, print a status update.

        Args:
            s: string containing a status update.

        Returns:
            None.
        '''
        if self._verbose:
            print(s)


def test():
    ''' Test suite for some (but not yet all) of Hyperparameters'
    functionality.
    '''

    def test_helper(result, correct, name, verbose=False):
        if result == correct:
            print('Passed: %s.' % name)
        else:
            print('Failed %s.' % name)

        if verbose:
            print('\nResult:')
            print(result)

            print('\nCorrect:')
            print(correct)

    def test1():
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

        # Test integration of hps
        hps = Hyperparameters(inputs, defaults, non_hash)
        test_helper(hps.integrated_hps,
                    correct_integration,
                    'test 1a: simple nested hp integration with defaults')

        # Test partitioning for hashing
        test_helper(hps.hps_to_hash,
                    {'a': 'a2', 'c': 'c2'},
                    'test 1b: hp partitioning for hashing')

        # Test base case __getattr__ (no recursion necessary)
        test_helper(hps['a'],
                    'a2',
                    'test 1c: base __getattr__ (no recursion)')

        # Test recursive __getattr__
        test_helper(hps['b:c'],
                    'c2',
                    'test 1d: recursive __getattr__')

        # Test recursive __setattr__
        hps['e:g'] = 'g2'
        test_helper(hps['e:g'],
                    'g2',
                    'test 1e: recursive __setattr__')

        # Test recursive __setattr__ for originally non-existing item
        # (hopefully this never happens in practice)
        hps['e:h'] = 'h2'
        test_helper(hps['e:h'],
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
            'e': {'g': 'g2'}, # Complex hash override, want to only override 'g', but not by simply replacing 'e'
            'j': {'l': 'l2'}  # Complex non-hash override, want to only override 'l', but not by simply replacing 'j'
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

        test_helper(hps.integrated_hps,
                    correct_integration,
                    'test 2a: complex hp integration, partial dict overrides.')

        # These will be wrong or throw access errors if implementation is wrong
        test_helper(hps['e:g'],
                    'g2',
                    'test 2b: recursive __getattr__') # should be 'g2'
        test_helper(hps['j:k'],
                    'k1',
                    'test 2c: recursive __getattr__') # should be 'k1'

        # Test partitioning for hashing
        # should include d and g but NOT l
        test_helper(hps.hps_to_hash,
                    {'d': 'd2', 'g': 'g2'},
                    'test 2d: hp partitioning for hashing')

    def test3():
        # Test integrate_hps using the sub/super application
        # See docstring for integrate_hps
        super_def_hps = {'a': 1, 'b': 2, 'c_hps': {'d': 3, 'e': 4}}
        sub_def_hps = {'b': 5, 'c_hps': {'d': 6}}
        correct_result = {'a': 1, 'b': 5, 'c_hps': {'d': 6, 'e': 4}}
        result = Hyperparameters.integrate_hps(super_def_hps, sub_def_hps)

        test_helper(result,
                    correct_result,
                    'test 3a: subclass/superclass hp integration')

        test_helper(super_def_hps,
                    {'a': 1, 'b': 2, 'c_hps': {'d': 3, 'e': 4}},
                    'test 3b: integration effects on original superclass hps')

        test_helper(sub_def_hps,
                    {'b': 5, 'c_hps': {'d': 6}},
                    'test 3c: integration effects on original subclass hps')

    test1()
    print()

    test2()
    print()

    test3()
    print()

''' To Do: Get things working again. Then, ...

Create hp_dict class to manage all dict-based functions with appropriate
support for recursions and possibly colon-delimited access:

    __str__ --> print_helper
    _getitem
    _setitem
    _sorted_str_from_dict

class hp_dict(dict):

'''