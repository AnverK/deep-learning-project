import os

attacks = ['fgsm', 'adv_gan', 'pgd']
adv_models = ['fgsm', 'adv_gan', 'pgd']

for attack in attacks:
    if attack == 'adv_gan':
        distill_modes = [False, True]
    else:
        distill_modes = [False]

    for distilled in distill_modes:
        for adv_model in adv_models:
            os.system(f'python evaluate_attack.py --attack {attack} --adv-model {adv_model} --attack-is-distilled {distilled}')

            if adv_model == 'adv_gan': 
                os.system(f'python evaluate_attack.py --attack {attack} --adv-model {adv_model} --attack-is-distilled {distilled} --defense-is-distilled {True}')