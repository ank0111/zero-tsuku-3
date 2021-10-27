from graphviz import Digraph
import numpy as np


def plot_dot_graph(output, verbose=False, graph_name='graph'):
    dg = Digraph(filename=graph_name, format='png')

    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    dg.attr('node', color='orange', style='filled')
    dg.node(str(id(output)), _label_var(output, verbose))
    while funcs:
        func = funcs.pop()
        dg.attr('node', color='lightblue', shape='box')
        dg.node(str(id(func)), func.__class__.__name__)
        dg.attr('node', color='orange', shape='ellipse')
        for x in func.inputs:
            dg.node(str(id(x)), _label_var(x, verbose))

            if x.creator is not None:
                add_func(x.creator)

        edges = [(str(id(func)), str(id(y()))) for y in func.outputs]
        edges += [(str(id(x)), str(id(func))) for x in func.inputs]
        dg.edges(edges)

    dg.view(cleanup=True)


def _label_var(v, verbose=False):
    label = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            label += ': '
        label += str(v.shape) + ' ' + str(v.dtype)
    return label


def sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not hasattr(axis, 'len'):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape
    gy = gy.reshape(shape)
    return gy


def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m
