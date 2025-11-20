import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'


preprocess_224 = transforms.Compose([
    transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])

preprocess_224_clip = transforms.Compose([
    transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.mean=self.mean.cuda()
        self.std=self.std.cuda()

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss



class ClipLoss():
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.criterion = nn.CosineSimilarity()
    def get_loss(self, pred, target):  
        if(pred.shape[1]==1): 
            imgpred = preprocess_224_clip(torch.repeat_interleave(pred,3,dim=1))
        else:
            imgpred = preprocess_224_clip(pred)
        
        if(target.shape[1]==1): 
            imgtarget = preprocess_224_clip(torch.repeat_interleave(target,3,dim=1))
        else:
            imgtarget = preprocess_224_clip(target)

        pred_features = self.model.encode_image(imgpred)

        with torch.no_grad():
            target_features = self.model.encode_image(imgtarget)

        loss = torch.mean(1-self.criterion(pred_features, target_features))

        return loss
    
    
    
def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

