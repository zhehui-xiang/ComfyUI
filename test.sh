
#!/usr/bin/env bash

for duration in 5 15; do
    python3 set_raylight_workload.py --duration $duration --json_file video_minimax_h3_t2v2.json
    export TEST="opt_GPU_duration_${duration}"
    bash run_headless_minimax_h3.sh 2>&1 | tee "$TEST.log"
    export TEST="no_opt_GPU_duration_${duration}"
    H3_FUSED=0 H3_INT8_ATTN=0 bash run_headless_minimax_h3.sh 2>&1 | tee "$TEST.log"
    for gpu in 1 2 4; do
        python3 set_raylight_workload.py --duration $duration --gpu $gpu --json_file video_minimax_h3_t2v_raylight_api.json
        export TEST="opt_GPU${gpu}_duration_${duration}"
        bash run_headless_minimax_h3_raylight.sh 2>&1 | tee "$TEST.log"
        export TEST="no_opt_GPU${gpu}_duration_${duration}"
        H3_FUSED=0 H3_INT8_ATTN=0 bash run_headless_minimax_h3_raylight.sh 2>&1 | tee "$TEST.log"
    done
done
