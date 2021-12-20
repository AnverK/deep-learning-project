class Config():
    NUM_WORKERS = 0

    TARGET_MODEL_BATCH_SIZE = 256
    TARGET_MODEL_FOLDER = 'target_model'

    ADV_GAN_BATCH_SIZE = 256
    ADV_GAN_FOLDER = 'adv_gan'

    USER = 'mboss'
    SCRATCH_PATH = f'/cluster/scratch/{USER}'
    LOGS_PATH = f'{SCRATCH_PATH}/dl_logs'

    APE_GAN_BATCH_SIZE = 128
    APE_GAN_FOLDER = 'ape_gan'
    APE_GAN_lr=2e-4
    APE_GAN_epochs=5
    APE_GAN_xi1=0.7
    APE_GAN_xi2=0.3
    APE_GAN_checkpoint=f"{LOGS_PATH}/{APE_GAN_FOLDER}/last.chkpt"
