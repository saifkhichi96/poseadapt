_base_ = [
    "./rtmpose-t_8xb32-10e_poseadapt-density-o5_256x192.py",
]


################################### Continual Learning Overrides #####################################
custom_hooks = [
    # CL Strategy: Less-Forgetful Learning (LFL)
    dict(type="LFLPlugin", alpha=0.5),
    # Evolution Plugin (required)
    dict(
        type="DefaultEvolutionPlugin",
        mode="last",
        model_cfgs={
            # model cfgs to be updated during evolution
            # in this case, model architecture remains unchanged
        },
    ),
]
