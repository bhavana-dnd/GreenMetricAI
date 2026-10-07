from greenmetricai.features import extract_code_metrics
def test_metrics():
    m=extract_code_metrics("def f(x):\n    for i in range(x):\n        if i: print(i)\n")
    assert m.loc>0 and m.num_functions==1 and m.loop_depth>=1
