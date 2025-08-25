
# encoding: utf-8
import argparse
import torch.utils.data as data
import torch
import torchvision.utils as vutils
from PIL import Image
import os
import torchvision.transforms as transforms
from PerceptFace import PerceptFace
os.environ['CUDA_VISIBLE_DEVICES'] ='0'


data_path='original_VGGFace_224/'
save_path='protected_VGGFace_224/'




class Test_data(data.Dataset):
    def __init__(self, data_path, mode='a'):
        super(Test_data, self).__init__()
        self.image_dir= data_path
        self.images= sorted(os.listdir(self.image_dir))
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.image_dir, self.images[index])))
        return img, self.images[index]

    def __len__(self):
        return self.length



def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--FFM_pretrain', type=str,default='pretrained_models/90000_net_G.pth')
    parser.add_argument("--Arc_path", type=str, default='pretrained_models/arcface_checkpoint.tar', help="run ONNX model via TRT")
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=32, help='# of sample images')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True)
    return parser.parse_args(args)
args = parse()
print(args)


train_dataset = Test_data(data_path)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=1,
    shuffle=False, drop_last=True
)

print('Training images:', len(train_dataset))

PerceptFace= PerceptFace(args)
# todo 预训练模型
PerceptFace.load('pretrained_models/MSE_new_all_loss_id_5_rec_5_wa_5_step_40000.pt')
PerceptFace.eval()

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
def mynorm(image):
    image_temp = image * imagenet_std
    return image_temp + imagenet_mean

def mynorm_2(image):
    image_temp = image / imagenet_std
    return image_temp - imagenet_mean

def mynorm_(image):
    image_temp = image - imagenet_mean
    return image_temp / imagenet_std

it = 1

PerceptFace.eval()
with torch.no_grad():
    for original_imgs,name in train_dataloader:
        # 保护人脸
        original_imgs = original_imgs.cuda() if args.gpu else original_imgs
        imgs_fake, _= PerceptFace.generate_protected(original_imgs)
        samples = [mynorm(imgs_fake)]
        names = name[0]
        filename2 = save_path + names
        vutils.save_image(mynorm(imgs_fake), filename2, nrow=1)
        it = it + 1





