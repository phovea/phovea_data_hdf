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
    load_stratification(rows, rowtype, name.split('/')[-1])

    if os.path.exists(name+'_desc.csv'): #table case
      h5.create_array(group, 'rows', rows)
      h5.set_node_attr(group, 'type', 'table')
      import csv
      with open(name+'_desc.csv','r') as cc:
        desc = dict()
        lookup = dict(uint8=tables.UInt8Col,uint16=tables.UInt16Col,uint32=tables.UInt32Col,
                      int8=tables.Int8Col,int16=tables.Int16Col,int32=tables.Int32Col,
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
            if 'deceased' in keys:
              column['colors'] = ['#e41a1b', '#377eb8', '#4c4c4c']
              column['names'] = ['Deceased', 'Living', 'NA']
            if 'male' in keys:
              column['colors'] = ['blue', 'red', '#4c4c4c']
              column['names'] = ['Male', 'Female', 'NA']
            t = tables.EnumCol(enum_, 'NA', base='uint8', pos=pos)
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
              missing = np.iinfo(getattr(np, row[2])).min
              print row[2], missing
              m = lambda x : missing if x == 'NA' or x == '' else int(x)
              column['type'] = 'int'
              column['missing'] = missing
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
              if col['type'] == 'int' and col['missing'] == v: #exclude missing value from range computation
                v = None
              old = col['range'][0]
              col['range'][0] = v if v is not None and (old is None or v < old) else old
              old = col['range'][1]
              col['range'][1] = v if v is not None and (old is None or v > old) else old
          entry.append()

      h5.set_node_attr(group,'columns',columns)

    elif os.path.exists(name+'_cols.csv'): #matrix case
      h5.set_node_attr(group, 'type', 'matrix')

      with open(name+'_cols.csv','r') as cc:
        l = cc.readline().split(';')
        coltype = l[1].strip()
        h5.set_node_attr(group, 'coltype', coltype)

        mtype = [m.strip() for m in cc.readline().split(';')[2:]]

      cols = np.loadtxt(name+'_cols.csv', dtype=np.string_, delimiter=';', skiprows=1, usecols=(1,))
      load_stratification(cols,coltype, name.split('/')[-1])
      print mtype[0],mtype

      if mtype[0] == 'float32':
        print 'float32'
        h5.set_node_attr(group, 'value', 'real')
        data = np.genfromtxt(f, dtype=np.float32, delimiter=';', missing_values='NaN', filling_values=np.NaN)
        data = data[...,0:data.shape[1]-1]
        h5.set_node_attr(group, 'range', [np.nanmin(data), np.nanmax(data)])
      elif mtype[0] == 'categorical':
        keys = mtype[1:]
        keys.remove('-2147483648')
        keys = sorted(map(int, keys))
        keys.append(-128)
        print keys
        h5.set_node_attr(group, 'value', 'categorical')
        h5.set_node_attr(group, 'categories', keys)
        if -2 in keys:
          # 0;-1;-2;2;1
          h5.set_node_attr(group, 'colors', ['#0571b0', '#92c5de', '#dcdcdc', '#eeb3bb', '#ca0020', '#4c4c4c'])
          h5.set_node_attr(group, 'names', ['Homozygous deletion', 'Heterozygous deletion', 'NORMAL', 'Low level amplification', 'High level amplification', 'Unknown'])
        elif 1 in keys:
          h5.set_node_attr(group, 'colors', ['#dcdcdc', '#ff0000', '#4c4c4c'])
          h5.set_node_attr(group, 'names', ['Not Mutated', 'Mutated', 'Unknown'])
        data = np.genfromtxt(f, dtype=np.int8, delimiter=';', missing_values='-2147483648', filling_values=-128)
        data = data[...,0:data.shape[1]-1]

      if coltype == 'TCGA_SAMPLE': #transpose
        data = np.transpose(data)
        h5.set_node_attr(group, 'rowtype' ,coltype)
        h5.set_node_attr(group, 'coltype' ,rowtype)

        h5.create_array(group, 'rows', cols)
        h5.create_array(group, 'cols', rows)
      else:
        h5.create_array(group, 'rows', rows)
        h5.create_array(group, 'cols', cols)
      h5.create_array(group, 'data', data)

    h5.flush()

  h5.close()


for f in glob.glob('/vagrant/_data/TCGA_KIRC*'):
  convert_it(f)