class Config():
    NUM_WORKERS = 4

    TARGET_MODEL_BATCH_SIZE = 256
    TARGET_MODEL_FOLDER = 'target_model'

    ADV_GAN_BATCH_SIZE = 256
    ADV_GAN_FOLDER = 'adv_gan'

    USER = 'osaeedi'
    SCRATCH_PATH = f'/cluster/scratch/{USER}'
    LOGS_PATH = f'{SCRATCH_PATH}/dl_logs'

    APE_GAN_BATCH_SIZE = 128
    APE_GAN_FOLDER = 'ape_gan'
    APE_GAN_lr = 5e-5
    APE_GAN_epochs = 5
    # APE_GAN_gen_loss_scale=0.9
    # APE_GAN_dis_loss_scale=0.02

    APE_GAN_gen_loss_scale = 0.7
    APE_GAN_dis_loss_scale = 0.3

    BABOON_BATCH_SIZE = 128
    BABOON_FOLDER = 'baboon'
