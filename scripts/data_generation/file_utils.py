"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 17 December 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no 

Save and load objects to/from files
"""

import os
import uuid
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as f:  # Overwrites any existing file.
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_well_config_and_data(config, data, dataset_version):
    obj = {
        'config': config,
        'data': data,
    }

    script_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_path, '../../data', dataset_version, 'dump')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    fn = str(uuid.uuid4())  # uuid4 could not say if it is thread safe so running loop
    while os.path.isfile(os.path.join(data_path, fn)):
        fn = str(uuid.uuid4())

    save_object(obj, os.path.join(data_path, fn))
