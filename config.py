class Config():
    NUM_WORKERS = 4

    TARGET_MODEL_BATCH_SIZE = 256
    TARGET_MODEL_FOLDER = 'target_model'

    TARGET_MODEL_WHITE_BOX_FOLDER = 'converted_secret'
    TARGET_MODEL_BLACK_BOX_FOLDER = 'converted_adv_trained'

    TARGET_MODEL_CKPT = 'model.ckpt'

    ADV_GAN_BATCH_SIZE = 256
    ADV_GAN_FOLDER = 'adv_gan'
    ADV_GAN_CKPT = 'last.ckpt'

    IS_BLACK_BOX = True
    IS_DISTILLED = False

    USER = 'mboss'
    SCRATCH_PATH = f'/cluster/scratch/{USER}'
    LOGS_PATH = f'{SCRATCH_PATH}/dl_logs'

    APE_GAN_BATCH_SIZE = 128
    
    APE_GAN_FOLDER = 'ape_gan'
    
    APE_GAN_lr = 5e-5
    APE_GAN_epochs = 5
    
    APE_GAN_gen_loss_scale=0.9
    APE_GAN_dis_loss_scale=0.02

    APE_GAN_CKPT = 'last.ckpt'

    BABOON_BATCH_SIZE = 128
    BABOON_FOLDER = 'baboon'
