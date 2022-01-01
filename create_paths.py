from config import Config


class CreatePaths:
    def __init__(self, adv_model='adv_gan'):
        super(CreatePaths, self).__init__()

        self.adv_model = adv_model

    def create_paths(self):
        target_model_folder = f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}'

        target_model_path = f'{target_model_folder}/{Config.TARGET_MODEL_WHITE_BOX_FOLDER}/{Config.TARGET_MODEL_CKPT}'
        if Config.IS_BLACK_BOX:
            target_model_path = f'{target_model_folder}/{Config.TARGET_MODEL_BLACK_BOX_FOLDER}/{Config.TARGET_MODEL_CKPT}'

        adv_model_folder = f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}'

        defense_model_folder = f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}'

        if self.adv_model == 'adv_gan':
            adv_model_folder = f'{adv_model_folder}/whitebox'
            defense_model_folder = f'{defense_model_folder}/adv_gan_whitebox'

            if Config.IS_BLACK_BOX:
                adv_model_folder = f'{adv_model_folder}/blackbox'
                defense_model_folder = f'{defense_model_folder}/adv_gan_blackbox'

            adv_model_folder = f'{adv_model_folder}/not_distilled'
            defense_model_folder = f'{defense_model_folder}_not_distilled'

            if Config.IS_DISTILLED:
                adv_model_folder = f'{adv_model_folder}/distilled'
                defense_model_folder = f'{defense_model_folder}_distilled'

        elif self.adv_model == 'fgsm':
            defense_model_folder = f'{defense_model_folder}/fgsm_whitebox'

            if Config.IS_BLACK_BOX:
                defense_model_folder = f'{defense_model_folder}/fgsm_blackbox'

        elif self.adv_model == 'pgd':
            defense_model_folder = f'{defense_model_folder}/pgd_whitebox'

            if Config.IS_BLACK_BOX:
                defense_model_folder = f'{defense_model_folder}/pgd_blackbox'

        return target_model_path, adv_model_folder, defense_model_folder
