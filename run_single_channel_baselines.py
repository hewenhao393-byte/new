import argparse
from baseline_analysis.pipeline import run_pipeline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=run_pipeline(args.source,args.output)
    print(result.to_string(index=False))

if __name__=="__main__":
    main()
