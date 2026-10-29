#!/bin/bash
# The adapters windows in order: w1 (tests, no-adapter checks); w3 (with adapters) only if the LoRA kernel tests
# passed; w2 (agentlong against origin/main). Each takes the lock on its own, so other jobs may run between them.
#   chain.sh OUT        (setsid nohup ... < /dev/null &)
H=$(cd "$(dirname "$0")" && pwd); O=$1
$H/window.sh w1 "tests and no-adapter checks" $H/w1.steps $O/w1 || exit
grep -q "all lora kernel tests passed" $O/w1/test-lora-kernel.log && $H/window.sh w3 "with adapters" $H/w3.steps $O/w3
$H/window.sh w2 "agentlong vs main" $H/w2.steps $O/w2
