import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json

import torchvision
from torchvision import transforms

import numpy as np
from PIL import Image
import imageio
import gc


from utils.utils import latent_to_image, oht_to_scalar_regression, get_lr, in_size, set_seed
from utils.data_utils import MultiResolutiontrainData
import argparse
from stylesketch_utils.stylesketch import SketchGenerator,Discriminator
from stylesketch_utils.prepare_stylegan import prepare_stylegan
from bisenet import evaluate as bisenet_eval

import cv2

from tqdm import tqdm
from non_leaking import augment, AdaptiveAugment
from loss import VGGPerceptualLoss,ClipLoss,d_r1_loss


device_ids = [0,1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter()

def safe_center(mask_mean, H, W, crop_size=256):
    """
    mask_mean: [B,2]
    return: safe i, j
    """
    # float preventing int32 overflow
    m = mask_mean.float()

    # filter illegal value (nagetive / NaN / beyond image)
    valid = (m[:,0] >= 0) & (m[:,0] < H) & (m[:,1] >= 0) & (m[:,1] < W)
    if valid.sum() == 0:
        # fallback to image center
        cy = H // 2
        cx = W // 2
    else:
        good = m[valid]
        cy = good[:,0].mean().item()
        cx = good[:,1].mean().item()

    # left up crop point 
    i = int(cy - crop_size // 2)
    j = int(cx - crop_size // 2)

    # clamp legal range
    i = max(0, min(i, H - crop_size))
    j = max(0, min(j, W - crop_size))

    return i, j

def parallelize(model):
    """
    Distribute a model across multiple GPUs.
    """
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model, device_ids=device_ids)
    return model


def prepare_data(args):
    g_all, avg_latent = prepare_stylegan(args['stylegan_checkpoint'],args['saved_latent'])
    latent_dir = str(args['annotation_image_latent_path'])
    latent_all=[]

    for i in range(1,1+args['max_training']):
        single_latent = latent_dir + str(i) + ".pt"
        try:
            name=single_latent
            single_latent = torch.load(single_latent)
            print(f"{name} found")
            latent_all += single_latent.cuda()
        except:
            print(f"{single_latent} not found")
            continue
    
    #change
    # inputlatentnum=0
    # inputlatentname=1
    # while inputlatentnum<args['max_training']:
    #     single_latent = latent_dir + str(inputlatentname) + ".pt"
    #     try:
    #         name=single_latent
    #         single_latent = torch.load(single_latent)
    #         print(f"{name} found")
    #         latent_all += single_latent.cuda()
    #         inputlatentnum+=1
    #         inputlatentname+=1
    #     except:
    #         print(f"{single_latent} not found")
    #         inputlatentname+=1
    #         continue

    sketch_list = []
    im_list = []
    num_data = len(latent_all)
    print("Latent num:", num_data)
    print("data processing ...")
    tempIntForIter=0
    while (len(im_list)<len(latent_all)):
        tempIntForIter+=1
         
        if len(im_list) > args['max_training']:
            break
        
        if args['train_data'][-1].isupper():
            name = '{0:03d}_{1}.PNG'.format(tempIntForIter,args['train_data'])
        else:
            name = '{0:03d}_{1}.png'.format(tempIntForIter,args['train_data'])
        
        #if not visible, move next number
        try:
            sketch_img = Image.open(os.path.join( args['annotation_sketch_path'] , name)).convert('L')
        except:
            print(f"Open {os.path.join( args['annotation_sketch_path'] , name)} failed")
            continue
        print(name)
        sketch_bw = np.array( sketch_img )
        
        sketch_list.append(sketch_bw)

        name = '{0:03d}_ori_sj.png'.format(tempIntForIter)
        im_name = os.path.join( args['annotation_sketch_path'], name)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))
    print("end of images ",tempIntForIter)

    all_skethes = np.stack(sketch_list)

    list_feature_maps =[]
    list_sketches=[]
    vis = []
    with torch.no_grad():
        for i in range(len(latent_all) ):

            gc.collect()
            latent_input = latent_all[i].float()

            img, feature_maps = latent_to_image(g_all, latent_input.unsqueeze(0), 3,dim=args['dim'][1],
                                                use_style_latents=args['annotation_data_from_w'])
            
            feature_maps.append(img.transpose(0,3,1,2))
            the_sketch = all_skethes[i]

            list_feature_maps.append(feature_maps)
            list_sketches.append(the_sketch.astype(np.float16))
       
            img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)
            mask3dim = np.stack((the_sketch,)*3, axis=-1)

            curr_vis = np.concatenate( [im_list[i], img_show, mask3dim], 0 )
            if(i<50):
                vis.append( curr_vis )


        vis = np.concatenate(vis, 1)
        imageio.imwrite(os.path.join(args['exp_dir'], "train_data.jpg"), vis)
    return list_feature_maps, list_sketches, num_data


def main(args):

    all_feature_maps_train_all, all_mask_train_all, num_data = prepare_data(args) # List( List (np.array, ....), List (np.array, ....), List (np.array, ....) , ...)
    train_data = MultiResolutiontrainData(all_feature_maps_train_all, torch.FloatTensor(all_mask_train_all))

    print(" *********************** Current number data " + str(num_data) + " ***********************")
    batch_size = args['batch_size']
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    for MODEL_NUMBER in range(args['model_num']):
        LAMBDA_GEN_full=1
        training_l1=True
        
        if(MODEL_NUMBER%7==3):
            LAMBDA_L1_else = 0
            LAMBDA_Clip_gts_full = 30
            LAMBDA_VGG_gts_full = 0.4
            LAMBDA_VGG_gts_comp=LAMBDA_VGG_gts_full
            LAMBDA_GEN_hair = LAMBDA_GEN_full
            LAMBDA_Clip_gts_hair = 0
            training_l1=True
        
        elif(MODEL_NUMBER%7==2):
            LAMBDA_L1_else = 0
            LAMBDA_Clip_gts_full = 30
            LAMBDA_VGG_gts_full = 0.4
            LAMBDA_VGG_gts_comp=LAMBDA_VGG_gts_full
            LAMBDA_GEN_hair = LAMBDA_GEN_full
            LAMBDA_Clip_gts_hair = 0
            training_l1=True
            
        elif(MODEL_NUMBER%7==1):
            LAMBDA_L1_else = 0
            LAMBDA_Clip_gts_full = 50
            LAMBDA_VGG_gts_full = 0.6
            LAMBDA_VGG_gts_comp=LAMBDA_VGG_gts_full
            LAMBDA_GEN_hair = LAMBDA_GEN_full
            LAMBDA_Clip_gts_hair = 0
            training_l1=True
            
        elif(MODEL_NUMBER%7==0):
            LAMBDA_L1_else = 0
            LAMBDA_Clip_gts_full = 120
            LAMBDA_VGG_gts_full = 1.2
            LAMBDA_VGG_gts_comp=LAMBDA_VGG_gts_full
            LAMBDA_GEN_hair = LAMBDA_GEN_full
            LAMBDA_Clip_gts_hair = 0
            training_l1=True
            
                        
        see_others=True
        
        
        
        LEARNINGRATE_G = 0.000014
        LEARNINGRATE_D = 0.000014

        
        print("MODEL NUM : " + str(MODEL_NUMBER) + " LAMBDA_GEN : " + str(LAMBDA_GEN_full) +"LAMBDA_GEN_full "+ str(LAMBDA_Clip_gts_full) +" LAMBDA_VGG : "+ str(LAMBDA_VGG_gts_full))
       
        
        torch.cuda.empty_cache()
        generator = SketchGenerator()
        generator = generator.cuda()
        generator = parallelize(generator)
        #import pdb; pdb.set_trace()
        
        


        model_parameters = filter(lambda p: p.requires_grad, generator.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"number of trainable parameters : {params}")
        print("Discriminator setting")
        discriminator = Discriminator()
        discriminator = discriminator.cuda()
        discriminator = parallelize(discriminator)
        
        discriminator_face = Discriminator()
        discriminator_face = discriminator_face.cuda()
        discriminator_face = parallelize(discriminator_face)
        
        discriminator_hair = Discriminator()
        discriminator_hair = discriminator_hair.cuda()
        discriminator_hair = parallelize(discriminator_hair)
        print("Loss setting")
        criterion = nn.L1Loss()
        loss_func_gan = nn.BCELoss()
        try:
            clip_model = ClipLoss()
            vgg_model = VGGPerceptualLoss()
        except:
            try:
                clip_model = ClipLoss()
                vgg_model = VGGPerceptualLoss()
            except:
                clip_model = ClipLoss()
                vgg_model = VGGPerceptualLoss()
        print("Optimizer setting")
        optimizer = optim.Adam(generator.parameters(), lr=LEARNINGRATE_G)
        optimizer_dis = optim.Adam(discriminator.parameters(), lr=LEARNINGRATE_D)
        optimizer_face_dis = optim.Adam(discriminator_face.parameters(), lr=LEARNINGRATE_D)
        optimizer_hair_dis = optim.Adam(discriminator_hair.parameters(), lr=LEARNINGRATE_D)
        
        #change
        start_iter = args.get('start_iteration', 0)

        if start_iter > 0:
            cppath=os.path.join(
                            args['exp_dir'], 
                            f"model_iter{start_iter}_number{MODEL_NUMBER}{opts['train_data']}.pth"
                        )
            checkpoint = torch.load(cppath)
            print("\n     Loading generator checkpoint:", cppath)

            state_dict=checkpoint['model_state_dict']
            new_state_dict={}
    
            #change
            for k,v  in state_dict.items():
                if k.startswith("module."):
                    name=k[7:]
                else:
                    name=k
                new_state_dict[name]=v
            #change
            generator.load_state_dict(new_state_dict,strict =True)
            ckpt_path = os.path.join(
                args['exp_dir'],
                f"other_iter{start_iter}_number{MODEL_NUMBER}{opts['train_data']}.pth"
            )
            print("\n     Loading other checkpoint:", ckpt_path)
            ckpt = torch.load(ckpt_path)


            discriminator.load_state_dict( ckpt['discriminator'] )
            discriminator_face.load_state_dict( ckpt['discriminator_face'] )
            discriminator_hair.load_state_dict( ckpt['discriminator_hair'] )

            optimizer.load_state_dict( ckpt['optimizer_g'] )
            optimizer_dis.load_state_dict( ckpt['optimizer_d'] )
            optimizer_face_dis.load_state_dict( ckpt['optimizer_df'] )
            optimizer_hair_dis.load_state_dict( ckpt['optimizer_dh'] )

            ada_aug_p = ckpt['ada_aug_p']
            ada_augment = ckpt['ada_augment']

            iteration = ckpt['iteration']
            start_epoch = ckpt['epoch']

        else:
            iteration = 0
            start_epoch = 0
            ada_aug_p=0
            ada_augment = torch.tensor([0.0, 0.0], device=device)




        generator.train()
        discriminator.train()
        discriminator_hair.train()
        discriminator_face.train()


        TOTAL_EPOCH=1100
        ada_aug_step = 0.6/22400
        optimizer.zero_grad()
        optimizer_dis.zero_grad()
        r_t_stat=0
        #1500 epoch w/o early stop  

        wantepoch=TOTAL_EPOCH-start_iter//7
        runedepoch=0
        for epoch in range(start_epoch,TOTAL_EPOCH):
            # runedepoch+=1
            # if runedepoch>wantepoch:
            #     break
            gc.collect()
            torch.cuda.empty_cache()

            batchnum=1

            for i,(X_batch, y_batch) in enumerate(train_loader):
                print(f"                 current epoch:  {epoch}")
                print("   batchnum:",batchnum)
                batchnum+=1
                gc.collect()
                ba_si = y_batch.size(0)
                
                patch = (16,16)

                real_label = torch.ones(ba_si,*(patch), requires_grad=False).to(device)
                fake_label = torch.zeros(ba_si,*(patch), requires_grad=False).to(device)
                # process data
                X_batch = [x.to(device).float() for x in X_batch]
                y_batch = (y_batch.to(device).float() /255).unsqueeze(1)
                
                # generator
                generator.zero_grad()
                y_pred = generator(X_batch)

                

                x_batch_img = torch.mean(X_batch[-1],dim=1).unsqueeze(1)/255
                
                print("bisenet input shape:", (X_batch[-1]/255).shape)

                hairmask, elsemask, l_eye_mean,r_eye_mean,nose_mean,lip_mean= bisenet_eval(X_batch[-1]/255)
                print("hairmask shape:",hairmask.shape)
                print("elsemask shape:", elsemask.shape)
                print("l_eye_mean shape:",l_eye_mean.shape)
                print("nose_mean shape:",nose_mean.shape)

                #change                
                # if hairmask.sum() == 0:                
                #     print("mask is empty")        
                # else:        
                #     print("mask has content")               
                # print("l_eye_mean:", l_eye_mean)              
                # print("Whether NaN existing in left eye mask:", torch.isnan(l_eye_mean).any())               
                # print("l_eye_mean shape:", l_eye_mean.shape)                 
                # H, W = 1024, 1024                  
                # if (l_eye_mean[:,0] < 0).any() or (l_eye_mean[:,0] >= H).any():                     
                #     print("Y out of range！")                  
                # if (l_eye_mean[:,1] < 0).any() or (l_eye_mean[:,1] >= W).any():                     
                #     print("X out of range！")



                enl_list=[l_eye_mean,r_eye_mean,nose_mean,lip_mean]
                
                i_leye =torch.mean(torch.transpose((enl_list[0]-128),1,0)[0].to(torch.float32)).to(torch.int32)
                j_leye =torch.mean(torch.transpose((enl_list[0]-128),1,0)[1].to(torch.float32)).to(torch.int32)
                i_reye =torch.mean(torch.transpose((enl_list[1]-128),1,0)[0].to(torch.float32)).to(torch.int32)
                j_reye =torch.mean(torch.transpose((enl_list[1]-128),1,0)[1].to(torch.float32)).to(torch.int32)
                i_nose =torch.mean(torch.transpose((enl_list[2]-128),1,0)[0].to(torch.float32)).to(torch.int32)
                j_nose =torch.mean(torch.transpose((enl_list[2]-128),1,0)[1].to(torch.float32)).to(torch.int32)
                i_lip = torch.mean(torch.transpose((enl_list[3]-128),1,0)[0].to(torch.float32)).to(torch.int32)
                j_lip = torch.mean(torch.transpose((enl_list[3]-128),1,0)[1].to(torch.float32)).to(torch.int32)

                #change
                # B1, C1, H1, W1 = y_pred.shape
                # i_leye, j_leye = safe_center(l_eye_mean, H1, W1)
                # i_reye, j_reye = safe_center(r_eye_mean, H1, W1)
                # i_nose, j_nose = safe_center(nose_mean, H1, W1)
                # i_lip,  j_lip  = safe_center(lip_mean, H1, W1)
                #change

                ic, jc, h, w = transforms.RandomCrop.get_params(y_pred, output_size=(256,256))
                
                enl=torch.randint(0, 4,(1,))[0]

                
                enl_list=[l_eye_mean,r_eye_mean,nose_mean,lip_mean]
                #return only hair and bg
                y_pred_hair = y_pred *hairmask.unsqueeze(1)
                y_batch_hair = y_batch *hairmask.unsqueeze(1)
                y_pred_hair +=elsemask.unsqueeze(1)
                y_batch_hair += elsemask.unsqueeze(1)

                x_batch_hair = x_batch_img*hairmask.unsqueeze(1)
                x_batch_hair += elsemask.unsqueeze(1)
                #return without hair
                y_pred_face = y_pred *elsemask.unsqueeze(1)
                y_batch_face = y_batch *elsemask.unsqueeze(1)
                y_pred_face += hairmask.unsqueeze(1)
                y_batch_face+= hairmask.unsqueeze(1)
                
                x_batch_face = x_batch_img*elsemask.unsqueeze(1)
                x_batch_face += hairmask.unsqueeze(1)
                #change 
                print("i_reye =", i_reye, "j_reye =", j_reye)
                print("r_eye_mean =", r_eye_mean)
                y_pred_leye = transforms.functional.crop(y_pred,i_leye,j_leye,256,256)
                y_batch_leye = transforms.functional.crop(y_batch,i_leye,j_leye,256,256)
                y_pred_reye = transforms.functional.crop(y_pred,i_reye,j_reye,256,256)
                y_batch_reye = transforms.functional.crop(y_batch,i_reye,j_reye,256,256)
                y_pred_nose = transforms.functional.crop(y_pred,i_nose,j_nose,256,256)
                y_batch_nose = transforms.functional.crop(y_batch,i_nose,j_nose,256,256)
                y_pred_lip = transforms.functional.crop(y_pred,i_lip,j_lip,256,256)
                y_batch_lip = transforms.functional.crop(y_batch,i_lip,j_lip,256,256)
                
                #change
                print("y_pred shape:",y_pred.shape)
                print("x barch hair shape:", x_batch_hair.shape)
                print("x_batch_img shape:",x_batch_img.shape)
                print("y_pred_face shape:",y_pred_face.shape)

                x_batch_aug, _ = augment(x_batch_img, ada_aug_p)
                x_batch_hair_crop_aug, _ = augment(x_batch_hair, ada_aug_p)
                x_batch_face_crop_aug, _ = augment(x_batch_face, ada_aug_p)
                
                y_pred_aug, _ = augment(y_pred, ada_aug_p)
                y_pred_hair_crop_aug, _ = augment(y_pred_hair, ada_aug_p)
                y_pred_face_crop_aug, _ = augment(y_pred_face, ada_aug_p)
                
                y_batch_aug, _ = augment(y_batch, ada_aug_p)
                y_batch_hair_crop_aug, _ = augment(y_batch_hair, ada_aug_p)
                y_batch_face_crop_aug, _ = augment(y_batch_face, ada_aug_p)
                
                              
                out_dis = discriminator(y_pred_aug,x_batch_aug).squeeze(1)
                out_dis_hair = discriminator(y_pred_hair_crop_aug,x_batch_hair_crop_aug).squeeze(1)
                out_dis_face = discriminator(y_pred_face_crop_aug,x_batch_face_crop_aug).squeeze(1)
                gen_loss_else = LAMBDA_GEN_full*loss_func_gan(out_dis, real_label) +LAMBDA_GEN_full*loss_func_gan(out_dis_hair, real_label) + LAMBDA_GEN_full*loss_func_gan(out_dis_face, real_label)
                
                #change
                print("before CLIP mem:", torch.cuda.memory_allocated()/1024/1024, "MB")
                
                gts_clip_loss_else = clip_model.get_loss(y_pred,y_batch) * LAMBDA_Clip_gts_full
                if LAMBDA_Clip_gts_hair>0:
                    gts_clip_loss_else += clip_model.get_loss(y_pred_hair,y_batch_hair) * LAMBDA_Clip_gts_hair
                
                #change
                print("before VGG mem:", torch.cuda.memory_allocated()/1024/1024, "MB")

                gts_vgg_loss_else =LAMBDA_VGG_gts_full* vgg_model(y_pred,y_batch) + LAMBDA_VGG_gts_comp*(vgg_model(y_pred_leye,y_batch_leye) \
                    +vgg_model(y_pred_reye,y_batch_reye) +vgg_model(y_pred_nose,y_batch_nose) +vgg_model(y_pred_lip,y_batch_lip) )/4
                                
                loss = gen_loss_else  +gts_clip_loss_else +  gts_vgg_loss_else
                
                
                if training_l1 and epoch<TOTAL_EPOCH//5:
                    loss+=criterion(y_pred,y_batch)*20
                loss.backward()


                optimizer.step()
                optimizer.zero_grad()

                discriminator.zero_grad()
                discriminator_hair.zero_grad()
                discriminator_face.zero_grad()
                
                
            
                
                out_else_dis_real = discriminator(y_batch_aug,x_batch_aug).squeeze(1) # real img
                real_loss = loss_func_gan(out_else_dis_real,real_label)
                out_else_dis_fake = discriminator(y_pred_aug.detach(),x_batch_aug).squeeze(1) # discriminate
                fake_loss = loss_func_gan(out_else_dis_fake,fake_label)
                
                out_hair_dis_real = discriminator_hair(y_batch_hair_crop_aug,x_batch_hair_crop_aug).squeeze(1)
                real_loss_hair = loss_func_gan(out_hair_dis_real,real_label)
                out_hair_dis_fake = discriminator_hair(y_pred_hair_crop_aug.detach(),x_batch_hair_crop_aug).squeeze(1)
                fake_loss_hair = loss_func_gan(out_hair_dis_fake,fake_label)
            
                out_face_dis_real = discriminator_face(y_batch_face_crop_aug,x_batch_face_crop_aug).squeeze(1)
                real_loss_face = loss_func_gan(out_face_dis_real,real_label)
                out_face_dis_fake = discriminator_face(y_pred_face_crop_aug.detach(),x_batch_face_crop_aug).squeeze(1)
                fake_loss_face = loss_func_gan(out_face_dis_fake,fake_label)
                

                d_loss = real_loss + fake_loss
                d_loss_hair =real_loss_hair + fake_loss_hair
                d_loss_face =real_loss_face + fake_loss_face
                   
                d_loss.backward()
                d_loss_hair.backward()
                d_loss_face.backward()
                
                
                #ada_aug
                if args['ada_augment']:
                    ada_augment += torch.tensor((torch.sign(out_else_dis_real-0.5).sum().item(),out_else_dis_real.shape[0]), device=device)
                
                    if ada_augment[1] > 127:
                        pred_signs, n_pred = ada_augment.tolist()

                        r_t_stat = pred_signs / (n_pred*patch[0]*patch[1])
                        if r_t_stat > 0.6:
                            sign = 1
                        else:
                            sign = -1

                        ada_aug_p += sign * ada_aug_step * n_pred*2
                        ada_aug_p = min(0.6, max(0, ada_aug_p))
                        ada_augment.mul_(0)
                
                            
                
                optimizer_dis.step()
                optimizer_face_dis.step()
                optimizer_hair_dis.step()
                
                optimizer_dis.zero_grad()
                optimizer_face_dis.zero_grad()
                optimizer_hair_dis.zero_grad()
                
                
                d_regularize = (i % 16 == 0) and args['d_regularize']
                
                if d_regularize:
                    y_batch.requires_grad = True
                    y_batch_hair.requires_grad = True
                    y_batch_face.requires_grad = True
                    
                    real_pred = discriminator(y_batch, x_batch_img)
                    real_pred_hair = discriminator_hair(y_batch_hair, x_batch_hair)
                    real_pred_face = discriminator_face(y_batch_face, x_batch_face)
                    
                    real_pred = real_pred.view(y_batch.size(0), -1)
                    real_pred_hair = real_pred_hair.view(y_batch_hair.size(0), -1)
                    real_pred_face = real_pred_face.view(y_batch_face.size(0), -1)
                    
                    real_pred = real_pred.mean(dim=1).unsqueeze(1)
                    real_pred_hair = real_pred_hair.mean(dim=1).unsqueeze(1)
                    real_pred_face = real_pred_face.mean(dim=1).unsqueeze(1)

                    r1_loss = d_r1_loss(real_pred, y_batch)
                    r1_loss_hair = d_r1_loss(real_pred_hair, y_batch_hair)
                    r1_loss_face = d_r1_loss(real_pred_face, y_batch_face)
                    
                    discriminator.zero_grad()
                    discriminator_hair.zero_grad()
                    discriminator_face.zero_grad()
                    
                    (r1_loss * 16 + 0 * real_pred[0]).backward()
                    (r1_loss_hair * 16 + 0 * real_pred_hair[0]).backward()
                    (r1_loss_face * 16 + 0 * real_pred_face[0]).backward()

                    optimizer_dis.step()
                    optimizer_hair_dis.step()
                    optimizer_face_dis.step()
                    
                iteration += 1
                
                # save interm
                if iteration % 25 ==0:
                    add_dict={"aug_p ":ada_aug_p,"genE ": gen_loss_else.item(),  " gts_clipE ": gts_clip_loss_else.item()," gts_vggE ": gts_vgg_loss_else.item(),  "disc ": d_loss.item()}
                    writer.add_scalars("sketch_split1_back_{}".format(MODEL_NUMBER), add_dict, iteration)
                if iteration % 100 == 0:
                    print("iter ",iteration,"Else : genE : ", gen_loss_else.item(),  " gts_clipE : ", gts_clip_loss_else.item()," gts_vggE : ", gts_vgg_loss_else.item(),"  disc : ", d_loss.item())
                    torchvision.utils.save_image(y_pred[0], 'img_during_training/{}img{}_gen_pred{}.png'.format(MODEL_NUMBER,iteration,opts['train_data']))
                    torchvision.utils.save_image(y_batch[0], 'img_during_training/{}img{}_ori_pred{}.png'.format(MODEL_NUMBER,iteration,opts['train_data']))
                    torchvision.utils.save_image(X_batch[-1][0]/255, 'img_during_training/{}img{}_ori_color.png'.format(MODEL_NUMBER,iteration))
                    print("ada_aug_p ",ada_aug_p," l1 ", criterion(y_pred,y_batch))
                if iteration % 100 == 0:
                    print('checkpoint, Epoch : ', str(epoch))

                    model_path = os.path.join(
                        args['exp_dir'], 
                        f"model_iter{iteration}_number{MODEL_NUMBER}{opts['train_data']}.pth"
                    )
                    other_path = os.path.join(
                        args['exp_dir'], 
                        f"other_iter{iteration}_number{MODEL_NUMBER}{opts['train_data']}.pth"
                    )
                    torch.save({'model_state_dict': generator.state_dict()}, model_path)

                    torch.save({
                        'iteration': iteration,
                        'epoch': epoch,

                        'discriminator': discriminator.state_dict(),
                        'discriminator_face': discriminator_face.state_dict(),
                        'discriminator_hair': discriminator_hair.state_dict(),

                        'optimizer_g': optimizer.state_dict(),
                        'optimizer_d': optimizer_dis.state_dict(),
                        'optimizer_df': optimizer_face_dis.state_dict(),
                        'optimizer_dh': optimizer_hair_dis.state_dict(),

                        'ada_aug_p': ada_aug_p,
                        'ada_augment': ada_augment,
                    }, other_path)
    

        writer.flush()

        model_path = os.path.join(args['exp_dir'], 'model_{}{}.pth'.format(MODEL_NUMBER,opts['train_data']))
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': generator.state_dict()},model_path)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, default = "experiments/pseudoface3.json")
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--save_final', type=str, default='False')
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--num_sample', type=int,  default=30)
    parser.add_argument('--train_data', type=str, default = 'Disney_sketch_MJ')
    parser.add_argument('--start_iteration', type=int, default = 0)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
        

    if args.exp_dir != "":
        opts['exp_dir'] = args.exp_dir

    opts['train_data'] = args.train_data
    opts['start_iteration']=args.start_iteration
    
    print("Opt", opts)
    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    main(opts)
