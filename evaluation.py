import datetime

import torch
from torchvision import models
from torchvision import transforms
import timm
import os
from torch.utils.data import Dataset, DataLoader
import tqdm
from PIL import Image
import torch.nn as nn
# import args
from utils.item_model import exrators

criterion = nn.CrossEntropyLoss()


import args

args = args.getArgs()

BATCH_SIZE = 10
test_model = [
    'vit_base_patch16_224',
    'pit_b_224', 'cait_s24_224', 'visformer_small', 'deit_base_patch16_224',
    'tnt_s_patch16_224', 'levit_256','convit_base',
    'swin_base_patch4_window7_224',
    'resnet50', 'resnet152', 'densenet201',
    'densenet169', 'vgg16', 'vgg19',
    'wide_resnet101_2', 'resnetv2_50x1_bitm'
]


item_model = exrators(item_model=test_model)

class MyDataset(Dataset):
    def __init__(self, data_path, adv_path):
        self.data_path = data_path
        self.adv_path = adv_path
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # self.normalize,
        ])
        self.datas = []
        with open(args.label_path, 'r') as f:
            for line in f.readlines():
                file, label = line.split()[:2]
                label = int(label)
                self.datas.append((file, label))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        image_file, label = self.datas[index]
        image = Image.open(os.path.join(self.data_path, image_file)).convert('RGB')
        image = self.transform(image)

        adv = Image.open(os.path.join(self.adv_path, image_file)).convert('RGB')
        adv_images = self.transform(adv)
        return adv_images, image, label, image_file


def model_attack(model, adv_loader, modename='name'):
    model.eval()
    with torch.no_grad():
        test_total, test_actual_total, test_success, test_acc = 0, 0, 0, 0
        tqdm_bar = tqdm.tqdm(adv_loader)
        for i, data in enumerate(tqdm_bar):
            adv_images, images, labels, files = data
            batchsize = labels.shape[0]
            test_total += batchsize
            adv_images = adv_images.cuda()
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            advoutputs = model(adv_images)
            _, predicted_clean = torch.max(outputs.data, 1)
            _, predicted_adv = torch.max(advoutputs.data, 1)
            for _id in range(batchsize):
                if predicted_clean[_id] == labels[_id]:
                    test_actual_total += 1
                    # if predicted_adv[_id] != labels[_id]:
                if predicted_adv[_id] != predicted_clean[_id]:
                # if predicted_adv[_id] != labels[_id]:
                    test_success += 1
                if predicted_adv[_id] == labels[_id]:
                    test_acc += 1
    with open(os.path.join('./', result_path), 'a') as f:
        f.write(f"|\t{modename.ljust(30, ' ')}\t" 
              f"|\t{str(round((test_actual_total / test_total) * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round((test_acc / test_total) * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round((test_success / test_total) * 100, 2)).ljust(8, ' ')}\t|" + "\n")

    return round((test_success / test_total) * 100, 2)


if __name__ == '__main__':
    total_asr = 0
    cnn_asr = 0
    vit_asr = 0
    result_path = args.result_path
    advdataset = MyDataset(args.datapath, args.adv_tgr)
    # get time
    current_time = datetime.datetime.now()

    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join('./', result_path), 'a') as f:
        f.write('======================{0}=={1}_{2}={3}={4}==================='
                .format(formatted_time, args.method, args.threshold, args.method2, args.adv_tgr) + '\n')
        f.write('-' * 75 + "\n")
        f.write('|\tModel name\t\t\t\t\t\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|' + "\n")
    adv_loader = DataLoader(advdataset, batch_size=BATCH_SIZE, num_workers=0)
    for i, name in enumerate(test_model):
        asr = model_attack(item_model[i], adv_loader, name)
        if i < 9:
            vit_asr += asr
        else:
            cnn_asr += asr

    avg_cnn_asr = cnn_asr / 8.0
    avg_vit_asr = vit_asr / 9.0
    avg_total_asr = (avg_cnn_asr + avg_vit_asr) / 2.0

    print("(CNN) AVG ASR: ", avg_cnn_asr)
    print("(ViT) AVG ASR: ", avg_vit_asr)
    print("ALL AVG ASR", avg_total_asr)
    with open(os.path.join('./', result_path), 'a') as f:
        f.write(f"|The ASR of ViT\t{avg_vit_asr}\t" + "\n")
        f.write(f"|The ASR of CNN\t{avg_cnn_asr}\t" + "\n")

        f.write(f"|The ASR of all models\t{avg_total_asr}\t" + "\n")
