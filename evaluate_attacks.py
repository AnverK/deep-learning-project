import os

attacks = ['fgsm', 'adv_gan', 'pgd']
adv_models = ['fgsm', 'adv_gan', 'pgd']

for attack in attacks:
    if attack == 'adv_gan':
        distill_modes = ['', '--attack-is-distilled']
    else:
        distill_modes = ['']

    for distill_mode in distill_modes:
        for adv_model in adv_models:
            os.system(f'python evaluate_attack.py '
                      f'--attack {attack} --adv-model {adv_model} {distill_mode}')

            if adv_model == 'adv_gan':
                os.system(f'python evaluate_attack.py '
                          f'--attack {attack} --adv-model {adv_model} {distill_mode} --defense-is-distilled')
