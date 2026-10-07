from dataclasses import dataclass, asdict
import ast, math
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze

FEATURES = [
    "cc","mi","hv","loc","depth_nesting","num_functions",
    "avg_function_length","comment_to_code_ratio","technical_debt_index",
    "api_call_density","loop_depth","branching_factor",
    "coupling_between_classes","num_parameters"
]

@dataclass
class CodeMetrics:
    cc: float; mi: float; hv: float; loc: int; depth_nesting: float
    num_functions: int; avg_function_length: float; comment_to_code_ratio: float
    technical_debt_index: float; api_call_density: float; loop_depth: float
    branching_factor: float; coupling_between_classes: float; num_parameters: float
    def to_dict(self): return asdict(self)

def _depth(tree, loop=False):
    controls = (ast.For, ast.AsyncFor, ast.While) if loop else (
        ast.If,ast.For,ast.AsyncFor,ast.While,ast.Try,ast.With,ast.AsyncWith,ast.Match)
    def walk(node,d=0):
        return max([d]+[walk(c,d+int(isinstance(c,controls))) for c in ast.iter_child_nodes(node)])
    return walk(tree)

def extract_code_metrics(code):
    tree=ast.parse(code)
    raw=analyze(code)
    blocks=cc_visit(code)
    cc=float(sum(x.complexity for x in blocks)) if blocks else 1.0
    hv=float(h_visit(code).total.volume or 0.0)
    mi=float(mi_visit(code,multi=True))
    funcs=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    lengths=[max(1,n.end_lineno-n.lineno+1) for n in funcs]
    loc=int(raw.sloc)
    comments=int(raw.comments)
    branches=sum(isinstance(n,(ast.If,ast.For,ast.While,ast.Try,ast.ExceptHandler,ast.IfExp,ast.Match))
                 for n in ast.walk(tree))
    calls=sum(isinstance(n,ast.Call) for n in ast.walk(tree))
    names=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name): names.add(n.value.id)
        elif isinstance(n,ast.Call) and isinstance(n.func,ast.Name): names.add(n.func.id)
    nesting=_depth(tree)
    loop_depth=_depth(tree,True)
    debt=100*(.35*min(cc/20,1)+.35*(1-min(max(mi,0),100)/100)+
               .15*min(math.log1p(loc)/math.log1p(5000),1)+.15*min(nesting/10,1))
    return CodeMetrics(
        cc,mi,hv,loc,float(nesting),len(funcs),
        float(sum(lengths)/max(len(lengths),1)),
        comments/max(loc,1),debt,calls/max(loc,1),float(loop_depth),
        branches/max(len(funcs),1),float(len(names)),
        float(sum(len(f.args.args) for f in funcs))
    )
