BASE_DIR="benchmark/protocols/domain_incremental/"

MEMORY=128G TIME=04:00:00 PARTITION="RTXA6000,RTXA6000-AV,A100-80GB,H100,H200,H200-AV,L40S,L40S-AV" CPUS=128 mmtrain \
    $BASE_DIR/lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_256x192.py \
    $BASE_DIR/lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_ewc_256x192.py \
    $BASE_DIR/lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_ewc-s_256x192.py \
    $BASE_DIR/lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_lfl_256x192.py \
    $BASE_DIR/lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_lwf_256x192.py \
    $BASE_DIR/lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_rwalk_256x192.py \
    $BASE_DIR/lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_pt_256x192.py \
    $BASE_DIR/lighting-ll/rtmpose-t_8xb32-10e_poseadapt-lighting-ll_si_256x192.py \
    $BASE_DIR/lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_256x192.py \
    $BASE_DIR/lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_ewc_256x192.py \
    $BASE_DIR/lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_ewc-s_256x192.py \
    $BASE_DIR/lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_lfl_256x192.py \
    $BASE_DIR/lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_lwf_256x192.py \
    $BASE_DIR/lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_rwalk_256x192.py \
    $BASE_DIR/lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_pt_256x192.py \
    $BASE_DIR/lighting-lle/rtmpose-t_8xb32-10e_poseadapt-lighting-lle_si_256x192.py \
    $BASE_DIR/lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_256x192.py \
    $BASE_DIR/lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_ewc_256x192.py \
    $BASE_DIR/lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_ewc-s_256x192.py \
    $BASE_DIR/lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_lfl_256x192.py \
    $BASE_DIR/lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_lwf_256x192.py \
    $BASE_DIR/lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_rwalk_256x192.py \
    $BASE_DIR/lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_pt_256x192.py \
    $BASE_DIR/lighting-llv/rtmpose-t_8xb32-10e_poseadapt-lighting-llv_si_256x192.py \
    $BASE_DIR/lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_256x192.py \
    $BASE_DIR/lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_ewc_256x192.py \
    $BASE_DIR/lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_ewc-s_256x192.py \
    $BASE_DIR/lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_lfl_256x192.py \
    $BASE_DIR/lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_lwf_256x192.py \
    $BASE_DIR/lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_rwalk_256x192.py \
    $BASE_DIR/lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_pt_256x192.py \
    $BASE_DIR/lighting-ms/rtmpose-t_8xb32-10e_poseadapt-lighting-ms_si_256x192.py
