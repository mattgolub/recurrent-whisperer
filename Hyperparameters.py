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
from copy import deepcopy
import hashlib
import cPickle
from numpy import sort
import yaml

class Hyperparameters(object):
    '''A general class for managing a model's hyperparameters, their default
    values, and hashes for organizing a many-model directory structure.

    Example usage:

    hps = Hyperparameters(
        <instance hps dict>,
        <default hash hps dict>,
        <default non-hash hps dict>)

    # You can now access an hp either by:
    hps.<hp name>
    # or
    hps[<hp name>]

    # You can update an hp either by:
    hps.<hp name> = <new value>
    hps[<hp name>] = <new value>

    # DO NOT UPDATE AN HP VIA CLASS INTERNALS:
    hps.__dict__[<hp name>] = <new value> # DO NOT DO THIS
    hps._integrated_hps[<hp name>] = <new value> # DO NOT DO THIS

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
            override default values. All keys must be also included in either
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

        # FUTURE WORK: separate the underlying data structure from the rest
        # of the class functionality.

        # Use deepcopy to make sure that no future interactions affect the
        # source of these defaults. This may be a bit overkill, since deepcopy
        # is also used in integrate_hps(), but better safe than sorry.
        self.update_hps = deepcopy(hps)
        self.default_hash_hps = deepcopy(default_hash_hps)
        self.default_non_hash_hps = deepcopy(default_non_hash_hps)

        self._validate_args(
            self.update_hps, self.default_hash_hps, self.default_non_hash_hps)

        self.default_hps = {}
        self.default_hps.update(self.default_hash_hps)
        self.default_hps.update(self.default_non_hash_hps)

        # dict containing all hyperparameters set to defaults, except as
        # overridden by values in hps.
        self._integrated_hps = self.integrate_hps(
            self.default_hps, self.update_hps)

        self._hash_len = hash_len
        self._verbose = verbose

        ''' Create a class attribute for every key in the _integrated_hps dict.
        These attributes are updated
        '''
        for key, val in self._integrated_hps.iteritems():
            # setattr(self, key, val)
            self.__setattr__(key, val, debug=False)

        # If this worked, at least it will throw an error if someone attempts
        # to update via hps.attr_name = val. Then can probably clean up by
        # dynamically adding a setter method...but it doesn't work yet.
        # for key in self._integrated_hps:
        #     setattr(self, key, property(lambda self: self._integrated_hps[key]))

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

        def parse_helper(parser, D, str_prefix=''):
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
                    parse_helper(parser, val,
                        str_prefix=str_prefix + str(key) + ':')

                else:
                    raise argparse.ArgumentTypeError('Default value must be '
                        'bool, int, float, str, dict, or None, but was %s '
                        '(key=%s).' % (type(val), key))

        parser = argparse.ArgumentParser(description=description)
        parse_helper(parser, default_hps)

        args = parser.parse_args()

        # This has no dicts (all values are bool, int, float, or str)
        hps_flat = vars(args)

        # As is, any values that were set to None at the command line will
        # now be 'None' in the dict. Here, change 'None' to None. This is less
        # easily implemented as a type argument function provided to
        # ArgumentParser (as is done with str2bool.)
        for key, val in hps_flat.iteritems():
            if val == 'None':
                hps_flat[key] = None

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

                _integrated_hps = integrate_hps(default_hps, input_hps)

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

                _integrated_hps = integrate_hps(super_hps, subclass_hps)

            Subclass hyperparameters will override defaults specified by this
            superclass, and possibly add new hyperparameters not present in
            the superclass.

        Args:
            hps: dict containing hyperparameters to be updated.

            update_hps: dict containing updates (overrides + additions) to hps.

        Returns:
            _integrated_hps: dict containing the integrated hyperparameters.
        '''

        # copy needed to prevent infinite recursive links between data
        # structures in recursive models. deepcopy may be overkill.
        _integrated_hps = deepcopy(hps)
        update_hps = deepcopy(update_hps)

        for key, val in update_hps.iteritems():

            if not isinstance(val, dict) or key not in _integrated_hps:
                # Base case
                _integrated_hps[key] = val
            else:
                # Recurse
                _integrated_hps[key] = Hyperparameters.integrate_hps(
                    _integrated_hps[key], val)

        return _integrated_hps

    @property
    def hps_to_hash(self):
        return self._get_hash_hps(self._integrated_hps, self.default_hash_hps)

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
        return self._generate_hash(self._integrated_hps)

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
        file.write(cPickle.dumps(self._integrated_hps))
        file.close()

    def save_yaml(self, save_path):
        # Note, this doesn't do well with numpy variables.
        # Ideally, make sure everything is bool, int, float, string, etc.
        self._maybe_print('Writing Hyperparameters YAML file.')
        with open(save_path, 'w') as yaml_file:
            yaml.dump(self._integrated_hps,
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

    @staticmethod
    def remove_list_hps(hps):
        # Recursive
        keys_to_del = []
        for key, val in hps.iteritems():
            if isinstance(val, list):
                # Base case
                print('\nDetected list in HPs (%s)--ignoring this HP.' % key)
                print('This will throw an error if the HP gets a command-line value.\n')
                keys_to_del.append(key)
            elif isinstance(val, dict):
                # Recursion
                Hyperparameters.remove_list_hps(val)

        for key in keys_to_del:
            del hps[key]

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
        return self._integrated_hps

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

        return Hyperparameters._getitem(self._integrated_hps, key)

    def __setitem__(self, key, value):
        '''Assigns an individual hyperparameter value, with support for
        colon-delimited hyperparameter names. This is responsible for the
        functionality:

        hps = Hyperparameters(...)
        hps['hp_name'] = value

        Args:
            key: A string indicating the name of the hyperparameter to be
            assigned. Colon delimiting specifies traversals into
            (possibly nested) dicts within the master hyperparameters dict.
            Nested dicts are created if they do not already exist.

            value: The value of the hyperparameter to be assigned.

        Returns:
            None.
        '''

        # Update the value in self._integrated_hps
        Hyperparameters._setitem(self._integrated_hps, key, value)

        ''' Update the value stored as a class attribute. This may or may not
        be necessary given how original attributes were setup.'''
        if ':' in key:
            # Replace the entire class attribute dict with the updated one
            dict_name, rem_name = self._parse_delimited_hp_name(key)
            setattr(self, dict_name, self._integrated_hps[dict_name])
        else:
            setattr(self, key, value)

    def __str__(self):
        return self.print_sorted_dict(self._integrated_hps)

    # def __getattr__(self, name):
    #     ''' Failed attempt to link each key in _integrated_hps to a class
    #     attribute. This just ignites an infinite recursion.
    #     '''
    #     if name in self._integrated_hps:
    #         return self._integrated_hps[name]
    #     else:
    #         return super(Hyperparameters, self).__getattr__(name)

    def __setattr__(self, name, value, debug=True):
        ''' Updates _integrated_hps if name is a top-level HP key.

        Note: this does not support comma delimited name. This is generally not
        a problem because the following simply does not parse:

            hps.comma:delimited:hp:name = value

        But, the following does parse. This is not supported and will throw an
        error.

            hps.__setattr__('comma:delimited:hp:name', value) # AssertionError

        '''
        if hasattr(self, '_integrated_hps'):

            assert ':' not in name, \
                ('Comma-delimited name not supported in '
                'Hyperparameters.__setattr__(name, value)')

            if name in self._integrated_hps:
                Hyperparameters._setitem(self._integrated_hps, name, value)

        super(Hyperparameters, self).__setattr__(name, value)

    @staticmethod
    def _getitem(D, key):
        ''' Helper function to support comma delimiting in key via recursive
        traversal of D.

        Args:
            D: dict.
            key: string.
        '''

        if ':' in key:

            dict_name, rem_name = \
                Hyperparameters._parse_delimited_hp_name(key)

            return Hyperparameters._getitem(D[dict_name], rem_name)

        else:

            return D[key]

    @staticmethod
    def _setitem(D, key, value):
        ''' Helper function to support comma delimiting in key via recursive
        traversal of D.

        ALWAYS use this instead of integrate_hps[key] = value

        Args:
            D: dict.
            key: string.
            value: anything.

        '''
        if ':' in key:

            dict_name, rem_name = \
                Hyperparameters._parse_delimited_hp_name(key)

            if not dict_name in D.keys():

                D[dict_name] = dict()

            Hyperparameters._setitem(D[dict_name], rem_name, value)

        else:

            D[key] = value

    # ************************************************************************
    # Internal helper functions **********************************************
    # ************************************************************************

    @classmethod
    def printable_str_from_dict(cls, D, n_indent=0):
        ''' Quick and dirty PrettyPrinting of a dict that conforms to the
        internal API for a Hyperparameters dict (i.e., keys as strings or
        dicts).
        '''

        S = ''
        indent = n_indent * '\t'
        for key in sort(D.keys()):
            value = D[key]

            S += '%s%s: ' % (indent, key)
            if isinstance(value, dict):
                S += '\n' + cls.printable_str_from_dict(value, n_indent+1)
            else:
                S += '%s\n' % str(value)

        return S

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
            error_msg = ('Attempted to override hyperparameter(s) '
                'with no defined defaults.\n')
            for violating_key in input_set - default_union:
                # if violating_key.startswith(cls._wild_card_prefix):
                #     pass
                # else:
                error_msg += str('\t\'%s\' is not a valid hyperparameter '
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
            # Here string conversions are compared, rather than the values.
            # This extra tolerance provides important invariance to hashes
            # regardless of whether or not parse_command_line() was used
            # (which parses values provided as strings)
            if str(val) != str(flat_defaults[key]):
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