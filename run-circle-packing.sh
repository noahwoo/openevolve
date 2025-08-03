#!/bin/zsh

# stage-1
# python ./openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py \
#   --config examples/circle_packing/config_phase_1.yaml \
#   --checkpoint examples/circle_packing/openevolve_output/checkpoints/checkpoint_40 \
#   --iterations 200

# stage-2
# python ./openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py \
#   --config examples/circle_packing/config_phase_2.yaml \
#   --checkpoint examples/circle_packing/openevolve_output/checkpoints/checkpoint_240 \
#   --iterations 200

python ./openevolve-run.py examples/circle_packing/initial_program.py examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 1
