#!/bin/zsh

# stage-1
python ./openevolve-run.py examples/fast_softmax/initial_program.py examples/fast_softmax/evaluator.py \
  --config examples/fast_softmax/config.yaml \
  --iterations 10
