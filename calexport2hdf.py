__author__ = 'Samuel Gratzl'


import tables
import numpy as np
import glob
import json
import os


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

def convert_it(base):
  h5 = tables.open_file(base+'.h5','w')

  def clean_name(name):
    n = name.lower().replace(' ','').replace('-','').replace('(','_').replace('.','_').replace(')','').split('/')[-1]
    if n[0].isdigit():
      n = '_' + n
    return n

  for f in glob.glob(base+'/*_data.csv'):
    name = f.replace('_data.csv','')
    cleaned = clean_name(name)
    print cleaned

    group = h5.create_group('/',cleaned, name.split('/')[-1])

    def load_stratification(ids, idtype, origin):
      if not os.path.exists(name+'_'+idtype+'.json'):
        return
      with open(name+'_'+idtype+'.json') as fs:
        strats = json.load(fs)
        for key,value in strats.iteritems():

          s = h5.create_group('/',clean_name(cleaned+'_'+key), origin+'/'+key)
          h5.set_node_attr(s, 'type', 'stratification')
          h5.set_node_attr(s, 'idtype', idtype)
          h5.set_node_attr(s, 'origin', origin)
          for gg,indices in value.iteritems():
            h5.create_array(s, clean_name(gg), ids[indices], gg)


    with open(name+'_rows.csv','r') as cc:
      l = cc.readline().split(';')
      rowtype = l[1].strip()
      h5.set_node_attr(group, 'rowtype', rowtype)

    rows = np.loadtxt(name+'_rows.csv', dtype=np.string_, delimiter=';', skiprows=1, usecols=(1,))
    h5.create_array(group, 'rows', rows)
    load_stratification(rows, rowtype, name.split('/')[-1])

    if os.path.exists(name+'_desc.csv'): #table case
      h5.set_node_attr(group, 'type', 'table')
      import csv
      with open(name+'_desc.csv','r') as cc:
        desc = dict()
        lookup = dict(uint8=tables.UInt8Col,uint16=tables.UInt16Col,uint32=tables.UInt32Col,
                      int8=tables.UInt8Col,int16=tables.Int16Col,int32=tables.Int32Col,
                      float16=tables.Float16Col,float32=tables.Float32Col,float64=tables.Float64Col,
                      bool=tables.BoolCol)
        columns = []
        mapper = []
        for i,row in enumerate(csv.reader(cc,delimiter=';')):
          if i == 0:
            continue
          t = None
          pos = int(row[0])
          column = dict(key=clean_name(row[1]),name=row[1])
          if row[2] == 'string':
            t = tables.StringCol(int(row[3]),pos=pos)
            column['type'] = 'string'
            m = str
          elif row[2] == 'categorical':
            keys = row[3:]
            keys.append('NA')
            print keys
            enum_ = tables.misc.enum.Enum(keys)
            column['type'] = 'categorical'
            column['categories'] = keys
            t = tables.EnumCol(enum_, 'NA', base='uint8',pos=pos)
            def wrap(e): #wrap in a function for the right scope
              return lambda x: e['deceased' if x == 'dead' else ('living' if x == 'alive' else x)]
            m = wrap(enum_)
          else:
            t2 = row[2]
            t = lookup[row[2]](pos=pos)
            if t2.startswith('float'):
              m = lambda x : np.NaN if x == 'NA' or x == '' else float(x)
              column['type'] = 'real'
            else:
              m = lambda x : -1 if x == 'NA' or x == '' else int(x)
              column['type'] = 'int'
            column['range'] = [None, None]
          desc[clean_name(row[1])] = t
          columns.append(column)
          mapper.append(m)

      table = h5.create_table(group,'table',desc)
      with open(name+'_data.csv','r') as d:
        entry = table.row
        for row in csv.reader(d,delimiter=';'):
          for col,m,v in zip(columns,mapper,row):
            v = m(v)
            entry[col['key']] = v
            if col['type'] == 'real' or col['type'] == 'int':
              old = col['range'][0]
              col['range'][0] = v if old is None or v < old else old
              old = col['range'][1]
              col['range'][1] = v if old is None or v > old else old
          entry.append()

      h5.set_node_attr(group,'columns',columns)

    elif os.path.exists(name+'_cols.csv'): #matrix case
      h5.set_node_attr(group, 'type', 'matrix')
      h5.set_node_attr(group, 'value', 'real')

      with open(name+'_cols.csv','r') as cc:
        l = cc.readline().split(';')
        coltype = l[1].strip()
        h5.set_node_attr(group, 'coltype', coltype)

      cols = np.loadtxt(name+'_cols.csv', dtype=np.string_, delimiter=';', skiprows=1, usecols=(1,))
      h5.create_array(group, 'cols', cols)
      load_stratification(cols,coltype, name.split('/')[-1])

      data = np.genfromtxt(f, dtype=np.float32, delimiter=';', missing_values='NaN', filling_values=np.NaN)
      data = data[...,0:data.shape[1]-1]
      h5.create_array(group, 'data', data)
      h5.set_node_attr(group, 'range', [np.nanmin(data), np.nanmax(data)])

    h5.flush()

  h5.close()


for f in glob.glob('/vagrant/_data/20*_GBM'):
  convert_it(f)