from config import Config


class CreatePaths:
    def __init__(
            self,
            adv_model='adv_gan',
            is_blackbox=Config.IS_BLACK_BOX,
            is_distilled=Config.IS_DISTILLED
    ):
        super(CreatePaths, self).__init__()

        self.adv_model = adv_model
        self.is_distilled = is_distilled
        self.is_blackbox = is_blackbox

    def create_paths(self):
        target_model_folder = f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}'

        if self.is_blackbox:
            target_model_path = f'{target_model_folder}/{Config.TARGET_MODEL_BLACK_BOX_FOLDER}/{Config.TARGET_MODEL_CKPT}'
        else:
            target_model_path = f'{target_model_folder}/{Config.TARGET_MODEL_WHITE_BOX_FOLDER}/{Config.TARGET_MODEL_CKPT}'

        adv_model_folder = f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}'
        defense_model_folder = f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}'

        if self.adv_model == 'adv_gan': 
            if self.is_blackbox:
                adv_model_folder = f'{adv_model_folder}/blackbox'
                defense_model_folder = f'{defense_model_folder}/adv_gan_blackbox'
            else:
                adv_model_folder = f'{adv_model_folder}/whitebox'
                defense_model_folder = f'{defense_model_folder}/adv_gan_whitebox'

            if self.is_distilled:
                adv_model_folder = f'{adv_model_folder}/distilled'
                defense_model_folder = f'{defense_model_folder}_distilled'
            else:
                adv_model_folder = f'{adv_model_folder}/not_distilled'
                defense_model_folder = f'{defense_model_folder}_not_distilled'

        elif self.adv_model == 'fgsm':
            if self.is_blackbox:
                defense_model_folder = f'{defense_model_folder}/fgsm_blackbox'
            else:
                defense_model_folder = f'{defense_model_folder}/fgsm_whitebox'

        elif self.adv_model == 'pgd':
            if self.is_blackbox:
                defense_model_folder = f'{defense_model_folder}/pgd_blackbox'
            else:
                defense_model_folder = f'{defense_model_folder}/pgd_whitebox'

        return target_model_path, adv_model_folder, defense_model_folder
