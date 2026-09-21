import argparse
from formal_model_selection.pipeline import run_pipeline

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--output",required=True); a=p.parse_args(); print(run_pipeline(a.source,a.output).to_string(index=False))
