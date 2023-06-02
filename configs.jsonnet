{
    BASE: {
        MODEL_ID: 'CODE_TEMPLATE', 
        USE_MULTI_GPU: false,
        SAME_PROB: 0,
        BATCH_PER_GPU: 2,
        MAX_STEP: 400000,
        SAVE_ROOT: 'train_results',
        PACKAGES_PATH: '../PACKAGES',
        DO_TRAIN: true,
        DO_VALID: true,
        DO_TEST: false,
        VAL_SIZE: 16,
        IMG_SIZE: 512,
        PORT: 3456
    },

    # weight of loss
    LOSS: {
        W_ADV: 1,
        W_VGG: 0,
        W_ID: 0,
        W_L1: 0,
        W_RECON: 0,
        W_CYCLE: 0,
        W_FEAT: 10,
        W_LPIPS: 10,
    },

    CYCLE: {
        LOSS: 10,
        IMAGE: 50,
        VALID: 50,
        TEST: 50,
        CKPT: 1000
    },

    CKPT: {
        # ckpt path
        # load checkpoints from ./train_result/{ckpt_id}/ckpt/G_{ckpt_step}.pt
        # if ckpt_id is empty, load G_latest.pt and D_latest.pt
        TURN_ON: false,
        ID_NUM: null,
        STEP: null,
    },

    WANDB: {
        TURN_ON: true,
        ALERT_THRES: 1000,
    },


    OPTIMIZER: {
        TYPE: 'Adam', # [Ranger, Adam]
        BETA: [0.0, 0.999], # default: Adam (0.9, 0.999) / Ranger (0.95, 0.999)
        LR_G: 0.0001,
        LR_D: 0.00001,
    },

    DATASET: {
        TRAIN_PATH:{
            IMAGE:
                [
                    '/data1/face-datasets/ffhq70k',
                ],
        },

        TEST_PATH:{
            IMAGE:
                [
                    './assets/images'
                ]
        }
    },




}