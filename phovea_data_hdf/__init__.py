###############################################################################
# Caleydo - Visualization for Molecular Biology - http://caleydo.org
# Copyright (c) The Caleydo Team. All rights reserved.
# Licensed under the new BSD license, available at http://caleydo.org/license
###############################################################################


def phovea(registry):
  """
  register extension points
  :param registry:
  """
  # generator-phovea:begin
  registry.append('dataset-provider', 'dataset-hdf', 'phovea_data_hdf.hdf', {})

  registry.append('json-encoder', 'dataset-hdf', 'phovea_data_hdf.json_encoder', {})

  registry.append('json-encoder', 'bytes-to-string-encoder', 'phovea_data_hdf.bytes_to_string_encoder', {})
  # generator-phovea:end
  pass


def phovea_config():
  """
  :return: file pointer to config file
  """
  from os import path
  here = path.abspath(path.dirname(__file__))
  config_file = path.join(here, 'config.json')
  return config_file if path.exists(config_file) else None
