'''
Hyperparameters.py
Version 1.0
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from copy import deepcopy
import hashlib
import cPickle
from numpy import sort
import yaml

class Hyperparameters(object):
    '''A general class for managing a model's hyperparameters, default
    settings, and hashes for organizing model checkpoint directories.

    '''
    def __init__(self,
                 hps=None,
                 default_hash_hps=None,
                 default_non_hash_hps=None,
                 hash_len=10):
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

            hash_len: int between 1 and 512 specifying the number of hex
            characters to include in the hyperparameters hash (all others are
            truncated). Larger values in this range may be necessary for
            massive hyperparameter searches, where the likelihood of hash
            collisions may be non-negligible. Default: 10.

        Returns:
            None.

        Raises:
            ValueError if at least one default_hash_hps or
            default_non_hash_hps is not specified.
        '''
        if default_hash_hps is None and default_non_hash_hps is None:
            raise ValueError('At lease one of default_hash_hps and '
                'default_non_hash_hps must be specified, but both were None.')

        self._hash_len = hash_len
        self._all_hps_as_dict, self._hash_hps_as_dict = \
            self._parse(hps, default_hash_hps, default_non_hash_hps)

        for key, val in self._all_hps_as_dict.iteritems():
            setattr(self, key, val)

    def __dict__(self):
        '''Returns hyperparameters in a dict.

        Args:
            None.

        Returns:
            dict of all hyperparameter names and settings as key, value pairs.
        '''
        return self._all_hps_as_dict

    def get_hash(self):
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
        return self._generate_hash(self._hash_hps_as_dict)[0:self._hash_len]

    def get_hash_all_hps(self):
        '''Computes a hash of all hyperparameters, including default and
        non-hash hyperparameters.

        Args:
            None.

        Returns:
            string containing the hyperparameters hash.
        '''
        return self._generate_hash(self._all_hps_as_dict)[0:self._hash_len]

    def _parse(self, input_hps, default_hash_hps, default_non_hash_hps):
        '''Parses the input arguments to __init__, returning a dict of all hps
        and a dict of hps to be hashed.

        Args:
            input_hps: See comment for hps in __init__.

            default_hash_hps: See comment for the same arg in __init__.

            default_non_hash_hps: See comment for the same arg in __init__.

        Returns:
            hps: dict containing all hyperparameters set to defaults, except
            as overridden by values in hps.

            hps_to_hash: dict containing the subset of hyperparameters in
            input_hps that are to be hashed (i.e., keys with non-default
            values that are also keys in default_hash_hps).

        Raises:
            ValueError if any key in input_hps is not included in exactly one
            of default_hash_hps or default_non_hash_hps.
        '''

        # Check to make sure each hp is either hash or non-hash (NOT BOTH)
        default_hash_hp_keys = default_hash_hps.keys()
        for non_hash_key in default_non_hash_hps.keys():
            if non_hash_key in default_hash_hp_keys:
                raise ValueError('Hyperparameter [%s] cannot be both hash '
                    'and non-hash.' % key)

        # Combine default hash and default non-hash hps
        default_hps = deepcopy(default_hash_hps)
        default_hps.update(default_non_hash_hps)

        # Initialize hps with default values
        hps = deepcopy(default_hps)
        hps_to_hash = dict()


        # Add non-default entries to dict for hashing
        for key, val in input_hps.iteritems():
            # Check to make sure all input_hps are valid
            # (i.e., have default values)
            # Otherwise, user may have mistyped an hp name.
            if not key in default_hps:
                raise ValueError('%s is not a valid hyperparameter '
                    '(has no specified default).' % key)
            else:
                # If valid, overwrite default values with input values
                hps[key] = val

                # If this hp should be hashed and value is not default, add it
                # to the dict for hashing
                if (key in default_hash_hp_keys) and (val != default_hps[key]):
                    hps_to_hash[key] = val

        return hps, hps_to_hash

    def _sorted_str_from_dict(self, d):
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
                str_val = self._sorted_str_from_dict(val)
            else:
                str_val = str(val)

            new_entry = (str(key) + ': ' + str_val)
            str_items.append(new_entry)

            if key_idx < n_keys:
                str_items.append(', ')

            key_idx += 1

        str_items.append('}')

        return ''.join(str_items)

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
        hps_hash = h.hexdigest()

        return hps_hash

    def write_yaml(self, save_path):
        # Note, this doesn't do well with numpy variables.
        # Ideally, make sure everything is bool, int, float, string, etc.
        print('Writing Hyperparameters YAML file.')
        with open(save_path, 'w') as yaml_file:
            yaml.dump(self._all_hps_as_dict,
                      yaml_file,
                      default_flow_style=False)

    def restore_from_yaml(self, yaml_path):
        # Note, this returns a dict of hps, not a Hyperparameters object
        # This is for use on runs created before save/restore was implemented
        # Preferred usage is with save/restore

        print('Reading Hyperparameters YAML file.')
        with open(yaml_path, 'r') as yaml_file:
            return yaml.load(yaml_file)

    def save(self, save_path):
        '''Saves the Hyperparameters object.

        Args:
            save_path: string containing the path at which to save these
            hyperparameter settings (including filename and arbitrary
            extension).

        Returns:
            None.
        '''
        print('Saving Hyperparameters.')
        file = open(save_path, 'wb')
        file.write(cPickle.dumps(self._all_hps_as_dict))
        file.close()

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
        print('Restoring Hyperparameters.')
        file = open(restore_path, 'rb')
        restore_data = file.read()
        file.close()
        return cPickle.loads(restore_data)
