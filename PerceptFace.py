
import torch
import torch.nn.functional as F

import torch.nn as nn
from fs_networks_fix import Generator_Adain_Upsample



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ID_transform(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=512):
        super(ID_transform, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层的全连接层
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 隐藏层到输出层的全连接层
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU激活函数
        self.fc3 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层的全连接层
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.fc3(x)
        return x


class PerceptFace(nn.Module):
    def __init__(self, args):
        super(PerceptFace, self).__init__()
        # self.lambda_id=args.lambda_id
        # self.lambda_wa = args.lambda_wa
        # self.lambda_rec=args.lambda_rec
        # Generator network
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9).to(device)
        self.netG.load_state_dict(torch.load(args.FFM_pretrain))
        self.netG.eval()

        self.netArc = torch.load(args.Arc_path, map_location=torch.device("cpu")).to(device)
        self.netArc.eval()


        self.ID_transformer=ID_transform(512).to(device)
        self.ID_transformer.eval()


    def generate_protected(self,img_a):
        with torch.no_grad():
            emb_img = self.netArc(F.interpolate(img_a, (112, 112), mode='bicubic'))
            original_id = F.normalize(emb_img, p=2, dim=1)
        T_id = self.ID_transformer(original_id)
        T_id = F.normalize(T_id, p=2, dim=1)
        ## 生成保护图像
        img_fake = self.netG(img_a, T_id)
        return img_fake,original_id


    def eval(self):
        self.ID_transformer.eval()


    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        self.ID_transformer.load_state_dict(states['WI'])





