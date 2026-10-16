#!/bin/bash
# trial.sh STUNT SEED -> /workspace/crabster/trials/STUNT_SEED
cd /workspace/repo-jumper-local-sim; source <(.venv/bin/python -c "from jhs.runner import gl_env; print('\n'.join(f'export {k}={v}' for k,v in gl_env().items()))")
O=/workspace/crabster/trials/$1_$2; mkdir -p $O
CB_DEBUG=${CB_DEBUG:-} CB_STUNT=$1 VS_SEED=$2 VS_OUT=$O VS_TMAX=${TMAX:-60} timeout 1500 .venv/bin/python jhs/crabster/run.py --app apps/jumper_surround_vision.app --scene /workspace/crabster/stage.map --backend native --device cpu --headless --steps 999999999 > $O/log.txt 2>&1
grep -E "^RESULT|Error" $O/log.txt | cut -c1-400
