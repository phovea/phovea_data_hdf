import tables


__author__ = 'Samuel Gratzl'


class TablesEncoder(object):
  def __contains__(self, obj):
    if isinstance(obj, tables.Array):
      return True
    return False

  def __call__(self, obj, base_encoder):
    if isinstance(obj, tables.Array):
      if obj.ndim == 1:
        return [x for x in obj]
      else:
        return [base_encoder.default(obj[i]) for i in range(obj.shape[0])]
    return None


n = TablesEncoder()


def create():
  return n
