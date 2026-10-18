"""Stand-in for `pygraphviz`, which `app.py` uses through `networkx.nx_agraph` to write graph.dot and
draw graph.svg. The container has no pygraphviz and nothing may be downloaded, but it has the
Graphviz `dot` binary, so this module records what networkx/app.py add (`AGraph`, `add_node`,
`add_edge`, `get_node(...).attr`, `write`, `draw`) and hands the DOT text to `dot` for the SVG.
"""

import shutil
import subprocess

__version__ = "0.0-shim"


def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


class _Node:
    def __init__(self, name):
        self.name = name
        self.attr = {}


class _Edge:
    def __init__(self, u, v, key=None):
        self.u, self.v, self.key = u, v, key
        self.attr = {}


class AGraph:
    def __init__(self, name="", strict=True, directed=False, **attrs):
        self.name = name
        self.strict = strict
        self.directed = directed
        self.graph_attr = dict(attrs)
        self.node_attr = {}
        self.edge_attr = {}
        self._nodes = {}
        self._edges = []

    def add_node(self, n, **attr):
        node = self._nodes.setdefault(str(n), _Node(str(n)))
        node.attr.update(attr)

    def add_edge(self, u, v, key=None, **attr):
        self.add_node(u)
        self.add_node(v)
        edge = _Edge(str(u), str(v), key)
        edge.attr.update(attr)
        self._edges.append(edge)

    def get_node(self, n):
        return self._nodes[str(n)]

    def get_edge(self, u, v, key=None):
        for e in reversed(self._edges):
            if e.u == str(u) and e.v == str(v) and (key is None or e.key == key):
                return e
        raise KeyError((u, v, key))

    def nodes(self):
        return list(self._nodes)

    def clear(self):
        self._nodes.clear()
        self._edges.clear()

    def close(self):
        pass

    def string(self):
        kind = "digraph" if self.directed else "graph"
        arrow = "->" if self.directed else "--"
        out = [f"{'strict ' if self.strict else ''}{kind} {_q(self.name) if self.name else ''} {{"]
        for scope, attrs in (("graph", self.graph_attr), ("node", self.node_attr), ("edge", self.edge_attr)):
            if attrs:
                out.append(f"\t{scope} [{', '.join(f'{k}={_q(v)}' for k, v in attrs.items())}];")
        for node in self._nodes.values():
            a = f" [{', '.join(f'{k}={_q(v)}' for k, v in node.attr.items())}]" if node.attr else ""
            out.append(f"\t{_q(node.name)}{a};")
        for e in self._edges:
            a = f" [{', '.join(f'{k}={_q(x)}' for k, x in e.attr.items())}]" if e.attr else ""
            out.append(f"\t{_q(e.u)} {arrow} {_q(e.v)}{a};")
        out.append("}")
        return "\n".join(out) + "\n"

    def write(self, path):
        with open(path, "w") as f:
            f.write(self.string())

    def draw(self, path, format=None, prog="dot", args=""):
        fmt = format or str(path).rsplit(".", 1)[-1]
        exe = shutil.which(prog)
        if exe is None:
            raise RuntimeError(f"Graphviz '{prog}' not found; cannot draw {path}")
        subprocess.run([exe, f"-T{fmt}", "-o", str(path)], input=self.string().encode(), check=True)
