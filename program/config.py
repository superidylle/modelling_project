# -*- coding: utf-8 -*-

import os

current_file = __file__

root_path = os.path.abspath(os.path.join(current_file, os.pardir, os.pardir))

input_data_path = os.path.abspath(os.path.join(root_path, 'data', 'input_data'))

output_data_path = os.path.abspath(os.path.join(root_path, 'data', 'output'))

