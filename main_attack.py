import os.path
import time

import torch
from tqdm import tqdm

from utils.get_dataset import get_dataset
from utils.get_models import get_models
from utils.tools import get_project_path
import argparse
from attack_method import MI_FGSM_SMER_meta
from torchvision.utils import save_image



def get_args():
    parser = argparse.ArgumentParser(description='NAMEA')
    parser.add_argument('--dataset', type=str, default='imagenet', help='imagenet')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='the batch size when training')
    parser.add_argument('--image-size', type=int, default=224,
                        help='image size of the dataloader')
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--attack_method', type=str, default='I_FGSM_SMER_meta')
    parser.add_argument('--image-dir', type=str, default='clean_img')
    parser.add_argument('--att-dir', type=str, default='')
    parser.add_argument('--image-info', type=str,
                        default='')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='gpu_id')
    # attack parameters
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=8.0)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=1.0,
                        help='default momentum value')
    args = parser.parse_args()
    return args


def main(args):
    cur_attack = {
                  'MI_FGSM_SMER_meta': MI_FGSM_SMER_meta,
                  }
    device = torch.device(f'cuda:{args.gpu_id}')
    # dataset
    dataloader = get_dataset(args)
    # models
    models, metrix = get_models(args, device=device)
    ens_model = ['resnet18', 'inc_v3', 'vit_t', 'deit_t']
    print(f'ens model: {ens_model}')
    ens_models = [models[i] for i in ens_model]
    total_success = [0] * 4
    total_num = 0
    total_time = 0
    start_time = time.time()
    for idx, (data, label, img_name) in enumerate(tqdm(dataloader)):
        n = label.size(0)
        data, label = data.to(device), label.to(device)
        adv_exp = cur_attack[args.attack_method](ens_models, data, label, args=args)
        pert = (adv_exp - data)

        print('L-infity:{}'.format((pert).max() / 255.))
        # adv_exp=data
        total_num += args.batch_size
        for model_name, model in models.items():
            with torch.no_grad():
                r_clean = model(data)
                r_adv = model(adv_exp)
            # clean
            pred_clean = r_clean.max(1)[1]
            correct_clean = (pred_clean == label).sum().item()
            # adv
            pred_adv = r_adv.max(1)[1]
            correct_adv = (pred_adv == label).sum().item()
            metrix[model_name].update(correct_clean, correct_adv, n)
        saved_dir = 'adv_dir'
        os.makedirs(saved_dir, exist_ok=True)
        for i in range(adv_exp.shape[0]):
            save_image(adv_exp[i], os.path.join(saved_dir, img_name[i].split('.')[0] + '.png'))
    total_time += (time.time() - start_time)
    print('-' * 73)
    print('|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    for model_name, _ in models.items():
        print(f"|\t{model_name.ljust(10, ' ')}\t"
              f"|\t{str(round(metrix[model_name].clean_acc * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round(metrix[model_name].adv_acc * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round(metrix[model_name].attack_rate * 100, 2)).ljust(8, ' ')}\t|")
    print('-' * 73)


if __name__ == '__main__':
    args = get_args()
    root_path = get_project_path()
    setattr(args, 'root_path', root_path)
    main(args)
