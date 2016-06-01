__author__ = 'sam'
import os
import numpy as np
import tables
import caleydo_server.range as ranges

import itertools

from caleydo_server.dataset_def import ADataSetEntry, ADataSetProvider

def assign_ids(ids, idtype):
  import caleydo_server.plugin

  manager = caleydo_server.plugin.lookup('idmanager')
  return np.array(manager(ids, idtype))

class HDFEntry(ADataSetEntry):
  def __init__(self, group, project):
    super(HDFEntry, self).__init__(group._v_title, project, group._v_attrs.type)
    self._group = group
    self._project = project
    self.path = self._group._v_pathname

class HDFMatrix(HDFEntry):
  def __init__(self, group, project):
    super(HDFMatrix, self).__init__(group, project)
    self._rowids = None
    self._colids = None
    self._range = None

  @property
  def rowtype(self):
    return self._group._v_attrs.rowtype

  @property
  def coltype(self):
    return self._group._v_attrs.coltype

  @property
  def value(self):
    return self._group._v_attrs.value

  @property
  def range(self):
    if 'range' in self._group._v_attrs:
      return self._group._v_attrs['range']
    if self._range is not None:
      return self._range
    d = self._group.data
    self._range = [np.nanmin(d), np.nanmax(d)]
    return self._range

  @property
  def shape(self):
    return self._group.data.shape

  def idtypes(self):
    return [self.rowtype, self.coltype]

  def to_description(self):
    r = super(HDFMatrix, self).to_description()
    r['rowtype'] = self.rowtype
    r['coltype'] = self.coltype
    r['value'] = v = dict(type=self.value)
    if self.value == 'real' or self.value == 'int':
      v['range'] = self.range
    if self.value == 'int' and hasattr(self._group._v_attrs,'missing'):
      v['missing'] = self._group._v_attrs.missing
    elif self.value == 'categorical':
      cats = self._group._v_attrs.categories
      if 'names' in self._group._v_attrs:
        v['categories'] = [ dict(name=cat,label=name,color=col) for cat,name,col in itertools.izip(cats,self._group._v_attrs.names,self._group._v_attrs.colors) ]
      else:
        v['categories'] = cats

    if 'center' in self._group._v_attrs:
      v['center'] = self._group._v_attrs['center']
    r['size'] = self.shape
    return r

  def mask(self, arr):
    if self.value == 'int' and hasattr(self._group._v_attrs,'missing'):
      missing = self._group._v_attrs.missing
      import numpy.ma as ma
      return ma.masked_equal(missing, arr)
    return np.array(arr)

  def aslist(self, range=None):
    return self.asnumpy(range)

  def asnumpy(self, range=None):
    n = self._group.data
    if range is None:
      return self.mask(n)
    rows = range[0].asslice()
    cols = range[1].asslice()
    d = None
    if isinstance(rows, list) and isinstance(cols, list):
      #fancy indexing in two dimension doesn't work
      d_help = n[rows,:]
      d = d_help[:,cols]
    else:
      d = n[rows, cols]


    if d.ndim == 1:
      #two options one row and n columns or the other way around
      if rows is Ellipsis or (isinstance(rows, list) and len(rows) > 1):
        d = d.reshape((d.shape[0],1))
      else:
        d = d.reshape((1,d.shape[0]))
    return self.mask(d)

  def rows(self, range=None):
    n = np.array(self._group.rows)
    if range is None:
      return n
    return n[range.asslice()]

  def rowids(self, range=None):
    if self._rowids is None:
      self._rowids = assign_ids(self.rows(), self.rowtype)
    n = self._rowids
    if range is None:
      return n
    return n[range.asslice()]

  def cols(self, range=None):
    n = np.array(self._group.cols)
    if range is None:
      return n
    return n[range.asslice()]

  def colids(self, range=None):
    if self._colids is None:
      self._colids = assign_ids(self.cols(), self.coltype)
    n = self._colids
    if range is None:
      return n
    return n[range.asslice()]


  def asjson(self, range=None):
    arr = self.asnumpy(range)
    rows = self.rows(None if range is None else range[0])
    cols = self.cols(None if range is None else range[1])
    rowids = self.rowids(None if range is None else range[1])
    colids = self.colids(None if range is None else range[1])

    r = dict(data=arr, rows=list(rows), cols=list(cols), rowIds=rowids, colids=colids)
    return r


class HDFVector(HDFEntry):
  def __init__(self, group, project):
    super(HDFVector, self).__init__(group, project)
    self._rowids = None
    self._range = None

  @property
  def idtype(self):
    return self._group._v_attrs.rowtype

  @property
  def value(self):
    return self._group._v_attrs.value

  @property
  def range(self):
    if 'range' in self._group._v_attrs:
      return self._group._v_attrs['range']
    if self._range is not None:
      return self._range
    d = self._group.data
    self._range = [np.nanmin(d), np.nanmax(d)]
    return self._range

  @property
  def shape(self):
    return len(self._group.data)

  def idtypes(self):
    return [self.idtype]

  def to_description(self):
    r = super(HDFVector, self).to_description()
    r['idtype'] = self.idtype
    r['value'] = dict(type=self.value,range=self.range)
    if self.value == 'int' and hasattr(self._group._v_attrs,'missing'):
      r['value']['missing'] = self._group._v_attrs.missing
    if 'center' in self._group._v_attrs:
      r['value']['center'] = self._group._v_attrs['center']
    r['size'] = [self.shape]
    return r

  def mask(self, arr):
    if self.value == 'int' and hasattr(self._group._v_attrs,'missing'):
      missing = self._group._v_attrs.missing
      import numpy.ma as ma
      return ma.masked_equal(missing, arr)
    return arr

  def aslist(self, range=None):
    return self.asnumpy(range)

  def asnumpy(self, range=None):
    n = self._group.data
    if range is None:
      return self.mask(n)
    return self.mask(n[range[0].asslice()])

  def rows(self, range=None):
    n = self._group.rows
    if range is None:
      return n
    return n[range.asslice()]

  def rowids(self, range=None):
    if self._rowids is None:
      self._rowids = assign_ids(self.rows(), self.rowtype)
    n = self._rowids
    if range is None:
      return n
    return n[range.asslice()]

  def asjson(self, range=None):
    arr = self.asnumpy(range)
    rows = self.rows(None if range is None else range[0])
    rowids = self.rowids(None if range is None else range[1])

    r = dict(data=arr, rows=rows, rowIds=rowids)
    return r


class HDFGroup(object):
  def __init__(self, name, range, color):
    self.name = name
    self.range = range
    self.color = color

  def __len__(self):
    return len(self.range)

  def dump(self):
    return dict(name=self.name, range=str(self.range), color=self.color)

def guess_color(name, i):
  name = name.lower()
  colors = dict(male='blue',female='red',deceased='#e41a1b',living='#377eb8')
  if name in colors:
    return colors[name]
  l = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
          '#ccebc5', '#ffed6f']
  return l[i%len(l)]

class HDFStratification(HDFEntry):
  def __init__(self, group, project):
    super(HDFStratification, self).__init__(group, project)
    self._rowids = None

  @property
  def idtype(self):
    return self._group._v_attrs.idtype

  def idtypes(self):
    return [self.idtype]

  def to_description(self):
    r = super(HDFStratification, self).to_description()
    r['idtype'] = self.idtype
    if 'origin' in self._group._v_attrs:
      r['origin'] = self._project + '/' + self._group._v_attrs.origin
    r['groups'] = [dict(name=gf._v_title, size=len(gf),color=gf._v_attrs['color'] if 'color' in gf._v_attrs else guess_color(gf._v_title, j))
                   for j,gf in enumerate(self._sortedgroup())]

    r['ngroups'] = len(r['groups'])
    r['size'] = [sum((g['size'] for g in r['groups']))]
    return r

  def _sortedgroup(self):
    return sorted(self._group._v_children.itervalues(), key=lambda x: x._v_title)

  def _rows(self):
    return np.concatenate(self._sortedgroup())

  def rows(self, range=None):
    n = self._rows()
    if range is None:
      return n
    return n[range[0].asslice()]

  def rowids(self, range=None):
    if self._rowids is None:
      self._rowids = assign_ids(self.rows(), self.idtype)
    n = self._rowids
    if range is None:
      return n
    return n[range.asslice()]

  def groups(self):
    i = 0
    for j,g in enumerate(self._sortedgroup()):
      name = g._v_title
      color = g._v_attrs['color'] if 'color' in g._v_attrs else guess_color(name, j)
      l = len(g)
      yield HDFGroup(name, ranges.from_slice(i, i + l), color)
      i += l

  def __getitem__(self, item):
    group = getattr(self._group, item)
    return group

  def asjson(self, range = None):
    r = dict(rows=self.rows(range), rowIds=self.rowids(range), groups=[g.dump() for g in self.groups()])
    return r


class HDFColumn(object):
  def __init__(self, attrs, group):
    self._group = group
    self.key = attrs['key']
    self.name = attrs['name']
    self.type = attrs['type']
    self._converter = lambda x: x
    if self.type == 'categorical':
      self._converter = tables.misc.enum.Enum(attrs['categories'])
      self.categories = attrs['categories']
    elif self.type == 'int' or self.type == 'real':
      self.range = attrs['range'] if 'range' in attrs else self.compute_range()
      self.missing = attrs.get('missing',None)

  def compute_range(self):
    d = self._group.table.col(self.key)
    return [np.nanmin(d), np.nanmax(d)]

  def __call__(self, v):
    return self._converter(v)


  def mask(self, arr):
    if self.type == 'int' and self.missing is not None:
      import numpy.ma as ma
      return ma.masked_equal(self.missing, arr)
    return arr

  def aslist(self, range=None):
    return self.asnumpy(range).tolist()

  def asnumpy(self, range=None):
    n = self._group.table.col(self.key)
    if range is not None:
      n = n[range[0].asslice()]
    return self.mask(n)

  def dump(self):
    value = dict(type=self.type)
    if self.type == 'categorical':
      value['categories'] = self.categories
    if self.type == 'int' or self.type == 'real':
      value['range'] = self.range
    if self.type == 'int' and self.missing is not None:
      value['missing'] = self.missing
    return dict(name=self.name, value=value)


class HDFTable(HDFEntry):
  def __init__(self, group, project):
    super(HDFTable, self).__init__(group, project)

    self.columns = [HDFColumn(a, group) for a in group._v_attrs.columns]
    self._rowids = None

  @property
  def idtype(self):
    return self._group._v_attrs.rowtype

  def idtypes(self):
    return [self.idtype]

  def to_description(self):
    r = super(HDFTable, self).to_description()
    r['idtype'] = self.idtype
    r['columns'] = [d.dump() for d in self.columns]
    r['size'] = [len(self._group.table), len(self.columns)]
    return r

  def rows(self, range=None):
    n = self._group.rows
    if range is None:
      return n
    return n[range.asslice()]

  def rowids(self, range=None):
    if self._rowids is None:
      self._rowids = assign_ids(self.rows(), self.idtype)
    n = self._rowids
    if range is None:
      return n
    return n[range.asslice()]

  def aslist(self, range=None):
    return self.aspandas(range).to_dict('records')

  def aspandas(self, range=None):
    import pandas as pd
    n = pd.DataFrame.from_records(self._group.table[:])
    if range is None:
      return n
    return n.iloc[range[0].asslice()]

  def filter(self, query):
    # perform the query on rows and cols and return a range with just the mathing one
    # np.argwhere
    np.arange(10)
    return ranges.all()

  def asjson(self, range=None):
    arr = self.aslist(range)
    rows = self.rows(None if range is None else range[0])
    rowids = self.rowids(None if range is None else range[0])

    dd = [ {c.key : c(row[c.key]) for c in self.columns } for row in arr]
    r = dict(data=dd, rows=rows, rowIds = rowids)

    return r


class HDFProject(object):
  def __init__(self, filename, baseDir):
    self.filename = filename
    p = os.path.relpath(filename, baseDir)
    project,_ = os.path.splitext(p)
    project = project.replace('.','_')
    self._h = tables.open_file(filename, 'r')

    self.entries = []
    for group in self._h.walk_groups('/'):
      if 'type' not in group._v_attrs:
        continue
      t = group._v_attrs.type
      if t == 'matrix':
        self.entries.append(HDFMatrix(group, project))
      elif t == 'stratification':
        self.entries.append(HDFStratification(group, project))
      elif t == 'table':
        self.entries.append(HDFTable(group, project))
      elif t == 'vector':
        self.entries.append(HDFVector(group, project))

  def __iter__(self):
    return iter(self.entries)

  def __len__(self):
    return len(self.entries)

  def __getitem__(self, dataset_id):
    for f in self.entries:
      if f.id == dataset_id:
        return f
    return None


class HDFFilesProvider(ADataSetProvider):
  def __init__(self):
    import caleydo_server.config
    c = caleydo_server.config.view('caleydo_data_hdf')
    from caleydo_server.util import glob_recursivly
    baseDir = caleydo_server.config.get('dataDir','caleydo_server')
    self.files = [HDFProject(f, baseDir) for f in glob_recursivly(baseDir,c.glob)]

  def __len__(self):
    return sum((len(f) for f in self.files))

  def __iter__(self):
    return itertools.chain(*self.files)

  def __getitem__(self, dataset_id):
    for f in self.files:
      r = f[dataset_id]
      if r is not None:
        return r
    return None


if __name__ == '__main__':
  # app.debug1 = True

  c = HDFFilesProvider()


def create():
  return HDFFilesProvider()
