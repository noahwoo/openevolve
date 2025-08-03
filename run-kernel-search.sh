#!/bin/zsh

# stage-1
python ./openevolve-run.py examples/kernel_search/initial_program.py examples/kernel_search/evaluator.py \
  --config examples/kernel_search/config.yaml \
  --iterations 200
