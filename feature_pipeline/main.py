import argparse,json
from pathlib import Path
from .pipeline import run_pipeline
from .validation import verify_output

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest"); p.add_argument("--workbook"); p.add_argument("--output",required=True); p.add_argument("--mode",choices=["record","temporal","both"],default="both"); p.add_argument("--limit",type=int); p.add_argument("--workers",type=int,default=1); p.add_argument("--verify-only",action="store_true"); a=p.parse_args()
    if a.verify_only: print(json.dumps(verify_output(a.output),ensure_ascii=False,indent=2)); return
    if not a.manifest or not a.workbook: p.error("--manifest and --workbook are required")
    modes=("record","temporal") if a.mode=="both" else (a.mode,)
    run_pipeline(Path(a.manifest),Path(a.workbook),Path(a.output),modes,a.limit,a.workers)
if __name__=="__main__": main()
