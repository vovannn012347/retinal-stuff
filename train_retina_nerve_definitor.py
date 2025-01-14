import os
import time
import cv2
import numpy as np

import utils.misc_retina as misc
from utils.misc_retina import magic_wand_mask_selection_faster, apply_clahe_rgb, apply_clahe_lab
from model.retina_classifier_networks import WeightedBCELoss, FcnskipNerveDefinitor2
from utils.retinaldata import ImageMaskDataset

import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF



parser = argparse.ArgumentParser(description='Glaucoma classify training')
parser.add_argument('--config', type=str,
                    default="configs/train_retina_glaucoma_definitor.yaml", help="Path to yaml config file")



def split_by_position(string, position):
    return [string[:position], string[position:]]

def training_loop(nerve_definitor_pass,  # convolution network
                  # glaucoma_definitor_pass,  # convolution network
                  optimizer,  # network optimizer
                  loss_func,  # network loss function
                  train_dataloader,  # training dataloader
                  last_n_iter,  # last iteration
                  config,  # Config object
                  ):
    device = torch.device('cuda' if torch.cuda.is_available()
                                    and config.use_cuda_if_available else 'cpu')

    init_process = True
    display_images = ''

    if config.display_iter:
        # Initialize an OpenCV window
        cv2.namedWindow("Image Processing", cv2.WINDOW_NORMAL)

    time0 = time.time()
    nerve_definitor_pass.train()
    init_n_iter = last_n_iter + 1
    train_iter = iter(train_dataloader)

    highest_difficulty_initialized = False
    highest_difficulty_images = []
    highest_difficulty_masks = []
    highest_difficulty_files = []
    highest_difficulty_losses = []
    lowest_loss = 0

    for n_iter in range(init_n_iter, config.max_iters):

        try:
            batch_real, batch_mask, batch_keys = next(train_iter)
        except Exception as e:
            train_iter = iter(train_dataloader)
            batch_real, batch_mask, batch_keys = next(train_iter)

        image_pass = torch.clone(batch_real).to(device)
        mask_pass = torch.clone(batch_mask).to(device)

        cv2.waitKey(1)

        image_pass = F.interpolate(image_pass, scale_factor=0.5, mode='bilinear', align_corners=False)
        mask_pass = F.interpolate(mask_pass, scale_factor=0.5, mode='bilinear', align_corners=False)

        output = nerve_definitor_pass(image_pass)

        loss_raw = loss_func(output, mask_pass)

        loss = torch.mean(loss_raw)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cv2.waitKey(1)

        if config.print_loss and (n_iter % config.print_loss == 0):
            # measure iterations/second
            dt = time.time() - time0
            print(f"@iter: {n_iter}: {(config.print_iter/dt):.4f} it/s")
            time0 = time.time()
            print(str(n_iter) + " iter loss: " + str(loss.item()))

        # high difficulty image pass
        if not highest_difficulty_initialized:
            highest_difficulty_images = torch.zeros_like(image_pass).to(device).copy_(image_pass)
            highest_difficulty_masks = torch.zeros_like(mask_pass).to(device).copy_(mask_pass)
            highest_difficulty_files = [batch_keys[i] for i in range(batch_keys.__len__())]
            highest_difficulty_losses = [torch.mean(loss_raw[i]).item() for i in range(loss_raw.size(0))]
            highest_difficulty_initialized = True
            lowest_loss = min(highest_difficulty_losses)
        else:
            # select highest losses and re-run on them every time
            current_losses = [torch.mean(loss_raw[i]).item() for i in range(loss_raw.size(0))]
            files_repeated_indexes = []

            # update losses with higher ones if possible
            for file_i in range(batch_keys.__len__()):
                try:
                    found_index = highest_difficulty_files.index(batch_keys[file_i])
                    files_repeated_indexes.append(found_index)
                except ValueError:
                    # Return -1 if the value is not found
                    found_index = -1

                if found_index > 0 and current_losses[file_i] > highest_difficulty_losses[found_index]:
                    # highest_difficulty_images[found_index] = image_pass[file_i]
                    # highest_difficulty_masks[found_index] = mask_pass[file_i]
                    highest_difficulty_losses[found_index] = current_losses[file_i]

            lowest_loss = min(highest_difficulty_losses)

            for file_i in range(batch_keys.__len__()):
                if files_repeated_indexes.__contains__(file_i):
                    continue

                # don't need to optimize this much for low batches
                if current_losses[file_i] > lowest_loss:
                    minloss_index = highest_difficulty_losses.index(min(highest_difficulty_losses))
                    highest_difficulty_images[minloss_index] = image_pass[file_i]
                    highest_difficulty_masks[minloss_index] = mask_pass[file_i]
                    highest_difficulty_losses[minloss_index] = current_losses[file_i]
                    highest_difficulty_files[minloss_index] = batch_keys[file_i]
                    lowest_loss = min(highest_difficulty_losses)

            # re-run optimizer for batch with largest losses
            image_pass = torch.clone(highest_difficulty_images).to(device)
            mask_pass = torch.clone(highest_difficulty_masks).to(device)

            output = nerve_definitor_pass(image_pass)

            loss_raw = loss_func(output, mask_pass)

            loss = torch.mean(loss_raw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update losses
            highest_difficulty_losses = [torch.mean(loss_raw[i]).item() for i in range(loss_raw.size(0))]
            lowest_loss = min(highest_difficulty_losses)

            if config.display_iter and (n_iter % config.display_iter == 0):

                #if init_process:
                display_images = image_pass

                display_output = nerve_definitor_pass(display_images)

                result_image1 = torch.cat([((display_images + 1) / 2)[i] for i in range(display_images.size(0))],
                                          dim=-1)
                '''torch.cat( 
                    [F.interpolate((display_images + 1) / 2, scale_factor=0.5, mode='bilinear', align_corners=False)[i]
                    for i in range(display_images.size(0))]
                    , dim=-1)'''
                result_image2 = torch.cat(
                    [display_output.expand(-1, 3, -1, -1)[i] for i in range(display_output.size(0))], dim=-1)
                result_image = torch.cat([result_image1, result_image2], dim=-2)
                result_image = result_image.detach().squeeze().permute(1, 2, 0).cpu().numpy()

                if init_process:
                    init_process = False
                    height, width = result_image.shape[:2]
                    cv2.resizeWindow("Image Processing", width // 2, height // 2)

                result_image = np.clip(result_image * 255, 0, 255).astype(np.uint8)
                cv2.imshow("Image Processing", result_image)
                cv2.waitKey(1)




        # todo: change optimizer values on the fly
        # loss_raw_values = [torch.mean(loss_raw.ite
        # for i in range(display_images.size(0))]


        # logging
        if config.print_iter and (n_iter % config.print_iter == 0):

            """result_image1 = torch.cat(
                [F.interpolate((image_pass + 1)/2, scale_factor=0.5, mode='bilinear', align_corners=False)[i]
                 for i in range(image_pass.size(0))], dim=-1)"""
            result_image1 = torch.cat([((image_pass + 1) / 2)[i] for i in range(image_pass.size(0))], dim=-1)
            result_image2 = torch.cat([output.expand(-1, 3, -1, -1)[i] for i in range(output.size(0))], dim=-1)
            result_image = torch.cat([result_image1, result_image2], dim=-2)
            img_out = TF.to_pil_image(result_image.squeeze().cpu(), mode="RGB")
            img_out.save(f"{config.log_dir}/training_images/iter_{n_iter}_mask.jpg")

        # save state dict snapshot
        if n_iter % config.save_checkpoint_iter == 0 \
                and n_iter > init_n_iter:
            misc.save_nerve_definitor("states.pth", nerve_definitor_pass, optimizer, n_iter, config)

        # save state dict snapshot backup
        if config.save_cp_backup_iter \
                and n_iter % config.save_cp_backup_iter == 0 \
                and n_iter > init_n_iter:
            misc.save_nerve_definitor(f"states_{n_iter}.pth", nerve_definitor_pass, optimizer, n_iter, config)


def main():

    args = parser.parse_args()
    config = misc.get_config(args.config)

    # set random seed
    if config.random_seed:
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        import numpy as np
        np.random.seed(config.random_seed)

    # make checkpoint folder if nonexistent
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(os.path.abspath(config.checkpoint_dir))
        os.makedirs(os.path.abspath(f"{config.checkpoint_dir}/images"))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    device_str = 'cuda' if torch.cuda.is_available() and config.use_cuda_if_available else 'cpu'

    device = torch.device(device_str)

    definitor = FcnskipNerveDefinitor2(num_classes=1)

    optimizer = torch.optim.Adam(definitor.parameters(),
                                 lr=config.opt_lr,
                                 betas=(config.opt_beta1, config.opt_beta2),
                                 weight_decay=config.weight_decay)

    transforms = [misc.RandomGreyscale(1),
                  T.RandomHorizontalFlip(0.5),
                  T.RandomVerticalFlip(0.5)]

    train_dataset = ImageMaskDataset(config.dataset_path,
                                     config.mask_folder_path,
                                     img_shape=config.img_shapes,
                                     scan_subdirs=config.scan_subdirs,
                                     transforms=transforms,
                                     device=device)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True)

    if config.gan_loss == 'wbce':
        train_loss = WeightedBCELoss(30, 100, reduction='none')
    elif config.gan_loss == 'bce':
        train_loss = nn.BCEWithLogitsLoss(weight=torch.tensor([30.0]), reduction='none')
    elif config.gan_loss == 'avg':
        train_loss = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    last_n_iter = -1

    if config.model_restore != '':
        state_dicts = torch.load(config.model_restore)
        definitor.load_state_dict(state_dicts['nerve_definitor'])
        if 'adam_opt_nerve_definitor' in state_dicts.keys():
            optimizer.load_state_dict(state_dicts['adam_opt_nerve_definitor'])
        last_n_iter = int(state_dicts['n_iter'])
        print(f"Loaded models from: {config.model_restore}!")
    else:
        definitor = FcnskipNerveDefinitor2.create_model()

    training_loop(definitor,
                  optimizer,
                  train_loss,
                  train_dataloader,
                  last_n_iter,
                  config)


if __name__ == '__main__':
    main()
