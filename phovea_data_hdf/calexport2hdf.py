"""
how to convert a caleydo project (without computation)
start hacked version and load the project
within org.caleydo.data the csv files will generated
rename the heterogeneous column csv file: xxx_cols.csv to xxx_desc.csv and edit it
1. remove header
2. fix name
3. add additional columns for the type: string (extra column the max length), int8, int16,int32, float16, float32, float64, enum (add as extra columns the categories)
convert it
"""

import tables
import numpy as np
import glob
import json
import os

__author__ = 'Samuel Gratzl'


def convert_it(base):
  h5 = tables.open_file(base + '.h5', 'w')

  def clean_name(name):
    n = name.lower().replace(' ', '').replace('-', '')
    n = n.replace('$', '').replace('(', '_').replace('.', '_')
    n = n.replace(')', '').split('/')[-1]

    if n[0].isdigit():
      n = '_' + n
    return n

  for f in glob.glob(base + '/*_data.csv'):
    name = f.replace('_data.csv', '')
    cleaned = clean_name(name)
    print(cleaned)

    group = h5.create_group('/', cleaned, name.split('/')[-1])

    def load_stratification(ids, idtype, origin):
      if not os.path.exists(name + '_' + idtype + '.json'):
        return None
      last = None, None
      with open(name + '_' + idtype + '.json') as fs:
        strats = json.load(fs)
        for key, value in strats.items():

          s = h5.create_group('/', clean_name(cleaned + '_' + key), origin + '/' + key)
          h5.set_node_attr(s, 'type', 'stratification')
          h5.set_node_attr(s, 'idtype', idtype)
          h5.set_node_attr(s, 'origin', origin)
          last = [], key
          for gg, indices in value.items():
            last[0].extend(indices)
            h5.create_array(s, clean_name(gg), ids[indices], gg)
      return last

    with open(name + '_rows.csv', 'r') as cc:
      line = cc.readline().split(';')
      rowtype = line[1].strip()
      h5.set_node_attr(group, 'rowtype', rowtype)

    rows = np.loadtxt(name + '_rows.csv', dtype=np.string_, delimiter=';', skiprows=1, usecols=(1,))
    default_row_strat, default_row_strat_name = load_stratification(rows, rowtype, name.split('/')[-1])

    if os.path.exists(name + '_desc.csv'):  # table case
      h5.create_array(group, 'rows', rows)
      h5.set_node_attr(group, 'type', 'table')
      import csv
      with open(name + '_desc.csv', 'r') as cc:
        desc = dict()
        lookup = dict(uint8=tables.UInt8Col, uint16=tables.UInt16Col, uint32=tables.UInt32Col,
                      int8=tables.Int8Col, int16=tables.Int16Col, int32=tables.Int32Col,
                      float16=tables.Float16Col, float32=tables.Float32Col, float64=tables.Float64Col,
                      bool=tables.BoolCol)
        columns = []
        mapper = []
        for i, row in enumerate(csv.reader(cc, delimiter=';')):
          if i == 0:
            continue
          t = None
          pos = int(row[0])
          column = dict(key=clean_name(row[1]), name=row[1])
          if row[2] == 'string':
            t = tables.StringCol(int(row[3]), pos=pos)
            column['type'] = 'string'
            m = str
          elif row[2] == 'categorical':
            keys = row[3:]
            if 'NA' not in keys:
              keys.append('NA')
            print(keys)
            enum_ = tables.misc.enum.Enum(keys)
            column['type'] = 'categorical'
            column['categories'] = keys
            if 'deceased' in keys:
              column['colors'] = ['#e41a1b', '#377eb8', '#4c4c4c']
              column['names'] = ['Deceased', 'Living', 'NA']
            if 'male' in keys:
              column['colors'] = ['blue', 'red', '#4c4c4c']
              column['names'] = ['Male', 'Female', 'NA']
            t = tables.EnumCol(enum_, 'NA', base='uint8', pos=pos)

            def wrap(e):  # wrap in a function for the right scope
              return lambda x: e['deceased' if x == 'dead' else ('living' if x == 'alive' else x)]

            m = wrap(enum_)
          else:
            t2 = row[2]
            t = lookup[row[2]](pos=pos)
            if t2.startswith('float'):

              def to_float(x):
                return np.NaN if x == 'NA' or x == '' else float(x)

              m = to_float
              column['type'] = 'real'
            else:
              missing = np.iinfo(getattr(np, row[2])).min
              print((row[2], missing))

              def to_int(x):
                return missing if x == 'NA' or x == '' else int(x)

              m = to_int
              column['type'] = 'int'
              column['missing'] = missing
            column['range'] = [None, None]
          desc[clean_name(row[1])] = t
          columns.append(column)
          mapper.append(m)

      table = h5.create_table(group, 'table', desc)
      with open(name + '_data.csv', 'r') as d:
        entry = table.row
        for row in csv.reader(d, delimiter=';'):
          for col, m, v in zip(columns, mapper, row):
            v = m(v)
            entry[col['key']] = v
            if col['type'] == 'real' or col['type'] == 'int':
              if col['type'] == 'int' and col['missing'] == v:  # exclude missing value from range computation
                v = None
              old = col['range'][0]
              col['range'][0] = v if v is not None and (old is None or v < old) else old
              old = col['range'][1]
              col['range'][1] = v if v is not None and (old is None or v > old) else old
          entry.append()

      h5.set_node_attr(group, 'columns', columns)

    elif os.path.exists(name + '_cols.csv'):  # matrix case
      h5.set_node_attr(group, 'type', 'matrix')

      with open(name + '_cols.csv', 'r') as cc:
        line = cc.readline().split(';')
        coltype = line[1].strip()
        h5.set_node_attr(group, 'coltype', coltype)

        mtype = [m.strip() for m in cc.readline().split(';')[2:]]

      cols = np.loadtxt(name + '_cols.csv', dtype=np.string_, delimiter=';', skiprows=1, usecols=(1,))
      default_col_strat, default_col_strat_name = load_stratification(cols, coltype, name.split('/')[-1])
      print(mtype)
      print((mtype[0]))

      if mtype[0] == 'float32':
        print('float32')

        with open(name + '_desc.json') as fs:
          stats = json.load(fs)
        h5.set_node_attr(group, 'value', 'real')
        with open(name + '_raw.csv') as fs:
          data = np.genfromtxt(fs, dtype=np.float32, delimiter='\t', missing_values='NaN', filling_values=np.NaN)
          # data = data[...,0:data.shape[1]]
          if stats['transposed']:
            data = np.transpose(data)
            # print
        min_v = stats.get('min', np.nanmin(data))
        max_v = stats.get('max', np.nanmax(data))
        data_raw = data
        data = data_raw.clip(min_v, max_v)
        h5.set_node_attr(group, 'range', [min_v, max_v])
        h5.set_node_attr(group, 'center', stats.get('center', 0))
      elif mtype[0] == 'int32':
        print('int32')
        h5.set_node_attr(group, 'value', 'int')
        missing = np.iinfo(np.int32).min
        h5.set_node_attr(group, 'missing', missing)
        data = np.genfromtxt(f, dtype=np.int32, delimiter=';', missing_values='NaN', filling_values=missing)
        data = data[..., 0:data.shape[1] - 1]
        h5.set_node_attr(group, 'range', [np.nanmin(data), np.nanmax(data)])
      elif mtype[0] == 'categorical':
        keys = mtype[1:]
        if '-2147483648' in keys:
          keys.remove('-2147483648')
        if 'UNKN@WN' in keys:
          keys.remove('UNKN@WN')
        keys = sorted(map(int, keys))
        keys.append(-128)
        print(keys)
        h5.set_node_attr(group, 'value', 'categorical')
        h5.set_node_attr(group, 'categories', keys)
        if -2 in keys:
          # 0;-1;-2;2;1
          h5.set_node_attr(group, 'colors', ['#0571b0', '#92c5de', '#dcdcdc', '#eeb3bb', '#ca0020', '#4c4c4c'])
          h5.set_node_attr(group, 'names',
                           ['Homozygous deletion', 'Heterozygous deletion', 'NORMAL', 'Low level amplification',
                            'High level amplification', 'Unknown'])
        elif 1 in keys:
          h5.set_node_attr(group, 'colors', ['#dcdcdc', '#ff0000', '#4c4c4c'])
          h5.set_node_attr(group, 'names', ['Not Mutated', 'Mutated', 'Unknown'])
        data = np.genfromtxt(f, dtype=np.int8, delimiter=';', missing_values={'-2147483648', 'UNKN@WN', 'NA', ''},
                             filling_values=-128)
        data = data[..., 0:data.shape[1] - 1]

      if coltype == 'TCGA_SAMPLE':  # transpose
        data = np.transpose(data)
        coltype, rowtype = rowtype, coltype
        rows, cols = cols, rows

        default_row_strat, default_col_strat = default_col_strat, default_row_strat
        default_row_strat_name, default_col_strat_name = default_col_strat_name, default_row_strat_name

      if default_col_strat and len(default_col_strat) == len(cols):
        print(('apply column stratification %s', default_col_strat_name))
        cols = cols[default_col_strat]
        data = data[:, default_col_strat]

      h5.set_node_attr(group, 'rowtype', rowtype)
      h5.set_node_attr(group, 'coltype', coltype)
      h5.create_array(group, 'rows', rows)
      h5.create_array(group, 'cols', cols)
      h5.create_array(group, 'data', data)

    h5.flush()

  h5.close()


for f in glob.glob('/vagrant/_data/calexport2hdf/*'):
  if os.path.isdir(f):
    convert_it(f)
