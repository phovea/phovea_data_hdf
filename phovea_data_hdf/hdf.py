
import os
import numpy as np
import tables
import phovea_server.range as ranges
import itertools
from phovea_server.dataset_def import ADataSetProvider, AColumn, AMatrix, AStratification, ATable, AVector

__author__ = 'sam'


def assign_ids(ids, idtype):
  import phovea_server.plugin

  manager = phovea_server.plugin.lookup('idmanager')
  return np.array(manager(ids, idtype))


def _resolve_categories(attrs):
  cats = attrs['categories']
  if isinstance(cats[0], str) or isinstance(cats[0], str):  # categories are strings
    converter = tables.misc.enum.Enum(cats)
    # create a numpy function out of it
    converter = np.vectorize(converter, otypes=['S' + str(max((len(c) for c in cats)))])
  else:
    converter = None
  if 'names' in attrs:
    names = attrs['names']
    colors = attrs['colors']
    if len(names) == len(cats) - 1:  # unknown missing
      names.insert(0, 'UNKN@WN')
      colors.insert(0, '#4c4c4c')
    cats = [dict(name=cat, label=name, color=col) for cat, name, col in
            zip(cats, names, colors)]

  return cats, converter


class HDFMatrix(AMatrix):
  def __init__(self, group, project):
    super(HDFMatrix, self).__init__(group._v_title, project, group._v_attrs.type.decode('utf-8'))
    self._group = group
    self._project = project
    self.path = self._group._v_pathname
    self._rowids = None
    self._colids = None
    self._range = None
    self.rowtype = self._group._v_attrs.rowtype.decode('utf-8')
    self.coltype = self._group._v_attrs.coltype.decode('utf-8')
    self.value = self._group._v_attrs.value.decode('utf-8')
    self.shape = self._group.data.shape
    if self.value == 'categorical':
      self.categories, self._converter = _resolve_categories(group._v_attrs)

  @property
  def range(self):
    if 'range' in self._group._v_attrs:
      return self._group._v_attrs['range']
    if self._range is not None:
      return self._range
    d = self._group.data
    self._range = [np.nanmin(d), np.nanmax(d)]
    return self._range

  def idtypes(self):
    return [self.rowtype, self.coltype]

  def to_description(self):
    r = super(HDFMatrix, self).to_description()
    r['rowtype'] = self.rowtype
    r['coltype'] = self.coltype
    r['value'] = v = dict(type=self.value)
    if self.value == 'real' or self.value == 'int':
      v['range'] = self.range
    if self.value == 'int' and hasattr(self._group._v_attrs, 'missing'):
      v['missing'] = self._group._v_attrs.missing.decode('utf-8')
    elif self.value == 'categorical':
      v['categories'] = self.categories

    if 'center' in self._group._v_attrs:
      v['center'] = self._group._v_attrs['center']
    r['size'] = self.shape
    return r

  def mask(self, arr):
    if self.value == 'int' and hasattr(self._group._v_attrs, 'missing'):
      missing = self._group._v_attrs.missing.decode('utf-8')
      import numpy.ma as ma
      return ma.masked_equal(arr, missing)

    if self.value == 'categorical' and self._converter is not None:
      return np.array(self._converter(arr))
    return np.array(arr)

  def asnumpy(self, range=None):
    n = self._group.data
    if range is None:
      return self.mask(n)
    rows = range[0].asslice()
    cols = range[1].asslice()
    d = None
    if isinstance(rows, list) and isinstance(cols, list):
      # fancy indexing in two dimension doesn't work
      d_help = n[rows, :]
      d = d_help[:, cols]
    else:
      d = n[rows, cols]

    if d.ndim == 1:
      # two options one row and n columns or the other way around
      if rows is Ellipsis or (isinstance(rows, list) and len(rows) > 1):
        d = d.reshape((d.shape[0], 1))
      else:
        d = d.reshape((1, d.shape[0]))
    elif d.ndim == 0:
      d = d.reshape((1, 1))
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


class HDFVector(AVector):
  def __init__(self, group, project):
    super(HDFVector, self).__init__(group._v_title, project, group._v_attrs.type.decode('utf-8'))
    self._group = group
    self._project = project
    self._rowids = None
    self._range = None
    self.idtype = self._group._v_attrs.rowtype.decode('utf-8')
    self.value = self._group._v_attrs.value.decode('utf-8')
    self.shape = len(self._group.data)

  @property
  def range(self):
    if 'range' in self._group._v_attrs:
      return self._group._v_attrs['range']
    if self._range is not None:
      return self._range
    d = self._group.data
    self._range = [np.nanmin(d), np.nanmax(d)]
    return self._range

  def idtypes(self):
    return [self.idtype]

  def to_description(self):
    r = super(HDFVector, self).to_description()
    r['idtype'] = self.idtype
    r['value'] = dict(type=self.value, range=self.range)
    if self.value == 'int' and hasattr(self._group._v_attrs, 'missing'):
      r['value']['missing'] = self._group._v_attrs.missing.decode('utf-8')
    if 'center' in self._group._v_attrs:
      r['value']['center'] = self._group._v_attrs['center']
    r['size'] = [self.shape]
    return r

  def mask(self, arr):
    if self.value == 'int' and hasattr(self._group._v_attrs, 'missing'):
      missing = self._group._v_attrs.missing.decode('utf-8')
      import numpy.ma as ma
      return ma.masked_equal(arr, missing)
    return arr

  def asnumpy(self, range=None):
    n = self._group.data
    if range is None:
      return self.mask(n)
    d = n[range[0].asslice()]
    if d.ndim == 0:
      d = d.reshape((1,))
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


class HDFGroup(object):
  def __init__(self, name, offset, data, color):
    self.name = name
    self.data = data
    self.range = ranges.from_slice(offset, offset + len(data))
    self.color = color

  def __len__(self):
    return len(self.data)

  def dump(self):
    return dict(name=self.name, range=str(self.range), color=self.color)

  def dump_desc(self):
    return dict(name=self.name, size=len(self), color=self.color)


def guess_color(name, i):
  name = name.lower()
  colors = dict(male='blue', female='red', deceased='#e41a1b', living='#377eb8')
  if name in colors:
    return colors[name]
  colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']
  return colors[i % len(colors)]


class HDFStratification(AStratification):
  def __init__(self, group, project):
    super(HDFStratification, self).__init__(group._v_title, project, group._v_attrs.type.decode('utf-8'))
    self._group = group
    self._project = project
    self._rowids = None
    self.idtype = self._group._v_attrs.idtype.decode('utf-8')
    self._groups = None

  def idtypes(self):
    return [self.idtype]

  def to_description(self):
    r = super(HDFStratification, self).to_description()
    r['idtype'] = self.idtype
    if 'origin' in self._group._v_attrs:
      r['origin'] = self._project + '/' + self._group._v_attrs.origin.decode('utf-8')
    r['groups'] = [g.dump_desc() for g in self.groups()]
    r['ngroups'] = len(r['groups'])
    r['size'] = [sum((g['size'] for g in r['groups']))]
    return r

  def _rows(self):
    return np.concatenate([g.data for g in self.groups()])

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
    if self._groups is None:
      self._groups = []
      i = 0
      values = iter(self._group._v_children.values())
      for j, g in enumerate(sorted(values, key=lambda x: x._v_title)):
        name = g._v_title
        color = g._v_attrs['color'] if 'color' in g._v_attrs else guess_color(name, j)
        length = len(g)
        self._groups.append(HDFGroup(name, i, g, color))
        i += length
    return self._groups

  def __getitem__(self, item):
    group = getattr(self._group, item)
    return group

  def asjson(self, range=None):
    r = dict(rows=self.rows(range), rowIds=self.rowids(range), groups=[g.dump() for g in self.groups()])
    return r


class HDFColumn(AColumn):
  def __init__(self, attrs, group):
    super(HDFColumn, self).__init__(attrs['name'], attrs['type'])
    self._group = group
    self.key = attrs['key']
    if self.type == 'categorical':
      self.categories, self._converter = _resolve_categories(attrs)
    elif self.type == 'int' or self.type == 'real':
      self.range = attrs['range'] if 'range' in attrs else self.compute_range()
      self.missing = attrs.get('missing', None)

  def compute_range(self):
    d = self._group.table.col(self.key)
    return [np.nanmin(d), np.nanmax(d)]

  def convert_category(self, vs):
    if self._converter is not None:
      return self._converter(vs)
    return vs

  def mask(self, arr):
    if self.type == 'int' and self.missing is not None:
      import numpy.ma as ma
      return ma.masked_equal(arr, self.missing)

    if self.type == 'categorical' and self._converter is not None:  # convert categorical columns
      return self._converter(arr)
    return arr

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
    return dict(name=self.name, value=value, column=self.key)


class HDFTable(ATable):
  def __init__(self, group, project):
    super(HDFTable, self).__init__(group._v_title, project, group._v_attrs.type.decode('utf-8'))
    self._group = group
    self._project = project

    self.columns = [HDFColumn(a, group) for a in group._v_attrs.columns]
    self._rowids = None
    self.idtype = self._group._v_attrs.rowtype.decode('utf-8')

  def idtypes(self):
    return [self.idtype]

  def to_description(self):
    r = super(HDFTable, self).to_description()
    r['idtype'] = self.idtype
    r['columns'] = [d.dump() for d in self.columns]
    r['size'] = [len(self._group.table), len(self.columns)]
    return r

  def rows(self, range=None):
    n = np.array(self._group.rows)
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

  def aspandas(self, range=None):
    import pandas as pd
    n = pd.DataFrame.from_records(self._group.table[:])
    # ensure right column order
    n = n[[item.key for item in self.columns]]

    # convert categorical enums
    # rename variable to avoid shadowing
    for item in self.columns:
      if item.type == 'categorical':
        n[item.key] = item.convert_category(n[item.key])

    if range is None:
      return n
    return n.iloc[range[0].asslice(no_ellipsis=True)]

  def filter(self, query):
    # perform the query on rows and cols and return a range with just the matching one
    # np.argwhere
    return ranges.all()


class HDFProject(object):
  def __init__(self, filename, base_dir):
    self.filename = filename
    p = os.path.relpath(filename, base_dir)
    project, _ = os.path.splitext(p)
    project = project.replace('.', '_')
    self._h = tables.open_file(filename, 'r')

    self.entries = []
    for group in self._h.walk_groups('/'):
      if 'type' not in group._v_attrs:
        continue
      t = group._v_attrs.type.decode('utf-8')
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
    from phovea_server import config
    # check initialization
    if config._c is None:
      config._initialize()
    conf = config.view('phovea_data_hdf')
    from phovea_server.util import glob_recursivly
    base_dir = config.get('dataDir', 'phovea_server')
    self.files = [HDFProject(f, base_dir) for f in glob_recursivly(base_dir, conf.get('glob'))]

  def __len__(self):
    return sum((len(f) for f in self))

  def __iter__(self):
    return iter((f for f in itertools.chain(*self.files) if f.can_read()))

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
