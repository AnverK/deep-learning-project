class Config():
    USER = 'akhusainov'
    SCRATCH_PATH = f'/cluster/home/{USER}/deep-learning-project'
    LOGS_PATH = f'{SCRATCH_PATH}/dl_logs'

    TARGET_MODEL_FOLDER = 'target_model'
    TARGET_MODEL_WHITE_BOX_FOLDER = 'converted_secret'
    TARGET_MODEL_BLACK_BOX_FOLDER = 'converted_adv_trained'
    TARGET_MODEL_CKPT = 'model.ckpt'

    ADV_GAN_BATCH_SIZE = 256
    ADV_GAN_FOLDER = 'adv_gan'
    ADV_GAN_CKPT = 'best.ckpt'

    IS_BLACK_BOX = True
    IS_DISTILLED = False

    APE_GAN_BATCH_SIZE = 64
    APE_GAN_FOLDER = 'ape_gan'
    APE_GAN_lr = 5e-5
    APE_GAN_epochs = 5
    APE_GAN_gen_loss_scale = 0.9
    APE_GAN_dis_loss_scale = 0.02
    APE_GAN_CKPT = 'best.ckpt'

    NUM_WORKERS = 4
