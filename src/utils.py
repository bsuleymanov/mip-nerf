def namedtuple_map(fn, tup):
    return type(tup)(*map(fn, tup))