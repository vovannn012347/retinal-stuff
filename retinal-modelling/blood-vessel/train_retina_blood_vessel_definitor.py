import os

import utils.misc_retina as misc
from model.retina_network import RetinalConvolutionNetwork
from model.retina_network import WeightedBCELoss, WeightedBCELoss2
from utils.retinaldata import RetinalTrainingDataset
from utils.misc_retina import gaussian_2d, magic_wand_mask_selection_batch_faster

import argparse
import torch.nn as nn
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(description='Blood vessel training')
parser.add_argument('--config', type=str,
                    default="configs/train_retina.yaml", help="Path to yaml config file")


def split_by_position(string, position):
    return [string[:position], string[position:]]


def training_loop(convolution_network_pass1,  # convolution network
                  convolution_network_pass2,  # convolution network
                  optimizer_pass1,  # network optimizer
                  optimizer_pass2,  # network optimizer
                  loss_func,  # network loss function
                  train_dataloader,  # training dataloader
                  last_n_iter,  # last iteration
                  config,  # Config object
                  ):
    device = torch.device('cuda' if torch.cuda.is_available()
                                    and config.use_cuda_if_available else 'cpu')

    convolution_network_pass1.train()
    convolution_network_pass2.train()

    init_n_iter = last_n_iter + 1
    train_iter = iter(train_dataloader)
    '''image_patch_size = config.img_patch_shapes'''
    '''image_patch_stride = config.img_stride
    gauss_patch = gaussian_2d([image_patch_size[0], image_patch_size[1]],
                              min_value=0.5, max_value=1.6, sigma=20)'''
    debug_save_img = False
    '''gauss_count = None'''

    for n_iter in range(init_n_iter, config.max_iters):

        try:
            batch_real, batch_mask = next(train_iter)
        except Exception as e:
            train_iter = iter(train_dataloader)
            batch_real, batch_mask = next(train_iter)

        image_pass = torch.clone(batch_real).to(device)
        mask_pass = torch.clone(batch_mask).to(device)

        '''channels, h, w = image_pass.shape

        image_patches_unfolded_pass1 = image_pass.unfold(1, image_patch_size[0], image_patch_stride[0])
        image_patches_unfolded_pass1 = image_patches_unfolded_pass1.unfold(2, image_patch_size[1],
                                                                           image_patch_stride[1])
        _, mask_unfold_count_h, mask_unfold_count_w, _, _ = image_patches_unfolded_pass1.size()
        image_patches_unfolded_pass1 = image_patches_unfolded_pass1.contiguous()
        image_patches_unfolded_pass1 = image_patches_unfolded_pass1.view(channels, -1,
                                                                         image_patch_size[0], image_patch_size[1])
        image_patches_unfolded_pass1 = image_patches_unfolded_pass1.permute(1, 0, 2, 3)
        
        output_pass1 = convolution_network_pass1(image_patches_unfolded_pass1)  # outputs sigmoid [0.0 to 1.0]
'''
        output_pass1 = convolution_network_pass1(image_pass) # outputs sigmoid [0.0 to 1.0]

        '''mask_for_test_unfolded_pass1 = mask_pass.unfold(1, image_patch_size[0], image_patch_stride[0])
        mask_for_test_unfolded_pass1 = mask_for_test_unfolded_pass1.unfold(2, image_patch_size[1],
                                                                           image_patch_stride[1])
        mask_for_test_unfolded_pass1 = mask_for_test_unfolded_pass1.contiguous()
        mask_for_test_unfolded_pass1 = mask_for_test_unfolded_pass1.view(1, -1, image_patch_size[0],
                                                                         image_patch_size[1])
        mask_for_test_unfolded_pass1 = mask_for_test_unfolded_pass1.permute(1, 0, 2, 3)'''

        mask1_pass2 = torch.zeros_like(output_pass1).to(device).copy_(output_pass1)

        '''image_patches_unfolded_pass2 = image_pass.unfold(1, image_patch_size[0], image_patch_stride[0])
        image_patches_unfolded_pass2 = image_patches_unfolded_pass2.unfold(2, image_patch_size[1],
                                                                           image_patch_stride[1])
        image_patches_unfolded_pass2 = image_patches_unfolded_pass2.contiguous()
        image_patches_unfolded_pass2 = image_patches_unfolded_pass2.view(channels, -1,
                                                                         image_patch_size[0], image_patch_size[1])
        image_patches_unfolded_pass2 = image_patches_unfolded_pass2.permute(1, 0, 2, 3)

        image_pass2 = torch.cat((image_patches_unfolded_pass2, (mask1_pass2 * 2 - 1)), dim=1)'''

        image_pass2 = torch.clone(batch_real).to(device)
        mask_pass2 = torch.clone(batch_mask).to(device)

        image_pass2 = torch.cat((image_pass2, (mask1_pass2 * 2 - 1)), dim=1)

        output_pass2 = convolution_network_pass2(image_pass2)  # outputs sigmoid [0.0 to 1.0]

        if (config.save_imgs_to_disc_iter and n_iter % config.save_imgs_to_disc_iter == 0) or debug_save_img:
            '''
            img_out = TF.to_pil_image(((image_pass + 1) / 2).squeeze().cpu(), mode="RGB")
            img_out.save(f"{config.log_dir}/training_images/iter_{n_iter}_source.png")
            
            image_stride_h = image_patch_stride[0]
            image_stride_w = image_patch_stride[1]

            #pass 1
            output_mask = output_pass1.permute(1, 0, 2, 3)
            reshaped_refolded_patches = output_mask.view(1, mask_unfold_count_h, mask_unfold_count_w,
                                                         image_patch_size[0], image_patch_size[1])

            pass_output1 = output_mask.view(output_mask.size(0), output_mask.size(1) * output_mask.size(2),
                                            output_mask.size(3))

            patches_output = image_patches_unfolded_pass1.permute(1, 0, 2, 3).view(
                                image_patches_unfolded_pass1.size(1),
                                image_patches_unfolded_pass1.size(0) * image_patches_unfolded_pass1.size(2),
                                image_patches_unfolded_pass1.size(3))

            merged_tensor = torch.zeros_like(mask_pass).to(device)
            if gauss_count is None:
                gauss_count = torch.zeros_like(mask_pass)

                for ch in range(reshaped_refolded_patches.size(0)):
                    for i in range(reshaped_refolded_patches.size(1)):
                        for j in range(reshaped_refolded_patches.size(2)):
                            x_start, y_start = i * image_stride_h, j * image_stride_w
                            gauss_count[ch,
                                        x_start:x_start + image_patch_size[0],
                                        y_start:y_start + image_patch_size[1]] += gauss_patch

            for ch in range(reshaped_refolded_patches.size(0)):
                for i in range(reshaped_refolded_patches.size(1)):
                    for j in range(reshaped_refolded_patches.size(2)):
                        x_start, y_start = i * image_stride_h, j * image_stride_w

                        # select all values
                        # temp_mask = reshaped_refolded_patches[ch, i, j] > 0.1

                        merged_tensor[
                            ch,
                            x_start:x_start + image_patch_size[0],
                            y_start:y_start + image_patch_size[1]] += \
                                (reshaped_refolded_patches[ch, i, j] * gauss_patch)
                            # (reshaped_refolded_patches[ch, i, j] * temp_mask * gauss_patch)

            merged_tensor /= gauss_count
            merged_tensor = torch.clamp_max(merged_tensor, 1)
            magic_wand_selected = magic_wand_mask_selection(merged_tensor).to(torch.float32)
                                                            # f"{config.log_dir}/training_images").to(torch.float32)

            # merged_tensor.mul_(0.5)
            result_image = torch.cat([magic_wand_selected, merged_tensor, mask_pass], dim=2)
            img_out = TF.to_pil_image(result_image.squeeze().cpu(), mode="L")
            img_out.save(f"{config.log_dir}/training_images/iter_{n_iter}_mask_pass1.png")

            #pass 2
            output_mask = output_pass2.permute(1, 0, 2, 3)
            reshaped_refolded_patches = output_mask.view(1, mask_unfold_count_h, mask_unfold_count_w,
                                                         image_patch_size[0], image_patch_size[1])

            pass_output2 = output_mask.view(output_mask.size(0), output_mask.size(1) * output_mask.size(2),
                                            output_mask.size(3))

            merged_tensor = torch.zeros_like(mask_pass).to(device)
            if gauss_count is None:
                gauss_count = torch.zeros_like(mask_pass)

                for ch in range(reshaped_refolded_patches.size(0)):
                    for i in range(reshaped_refolded_patches.size(1)):
                        for j in range(reshaped_refolded_patches.size(2)):
                            x_start, y_start = i * image_stride_h, j * image_stride_w
                            gauss_count[ch,
                                x_start:x_start + image_patch_size[0],
                                y_start:y_start + image_patch_size[1]] += gauss_patch

            for ch in range(reshaped_refolded_patches.size(0)):
                for i in range(reshaped_refolded_patches.size(1)):
                    for j in range(reshaped_refolded_patches.size(2)):
                        x_start, y_start = i * image_stride_h, j * image_stride_w

                        # select all values
                        # temp_mask = reshaped_refolded_patches[ch, i, j] > 0.1

                        merged_tensor[
                                ch,
                                x_start:x_start + image_patch_size[0],
                                y_start:y_start + image_patch_size[1]] += \
                                    (reshaped_refolded_patches[ch, i, j] * gauss_patch)
                                    # (reshaped_refolded_patches[ch, i, j] * temp_mask * gauss_patch)

            merged_tensor /= gauss_count
            merged_tensor = torch.clamp_max(merged_tensor, 1)
            magic_wand_selected = magic_wand_mask_selection(merged_tensor).to(torch.float32)
            # f"{config.log_dir}/training_images").to(torch.float32)'''

            image_rgb = torch.cat([((image_pass + 1) / 2)[i] for i in range(image_pass.size(0))],  dim=-1)

            #img_out = TF.to_pil_image(, mode="RGB")
            #img_out.save(f"{config.log_dir}/training_images/iter_{n_iter}_source.png")

            magic_wand_selected = magic_wand_mask_selection_batch_faster(output_pass2, lower_multipleir=0.15).to(torch.float32)

            mask_result = torch.cat([mask_pass.repeat(1, 3, 1, 1),
                                     output_pass1.repeat(1, 3, 1, 1),
                                     output_pass2.repeat(1, 3, 1, 1),
                                     magic_wand_selected.repeat(1, 3, 1, 1)], dim=-2)

            mask_result = torch.cat([mask_result[i] for i in range(image_pass.size(0))],  dim=-1)

            result_image = torch.cat(
                [image_rgb, mask_result], dim=-2)

            img_out = TF.to_pil_image(result_image.squeeze().cpu(), mode="RGB")
            img_out.save(f"{config.log_dir}/training_images/iter_{n_iter}.jpg")

            '''result_image = torch.cat([pass_output1.repeat(3, 1, 1), pass_output2.repeat(3, 1, 1), (patches_output + 1)/2], dim=2)
            img_out = TF.to_pil_image(result_image.squeeze().cpu(), mode="RGB")
            img_out.save(f"{config.log_dir}/training_images/iter_{n_iter}_long_patch.png")'''

        '''loss1 = loss_func[0](output_pass1, mask_for_test_unfolded_pass1)'''
        loss1 = loss_func[0](output_pass1, mask_pass)
        optimizer_pass1.zero_grad()

        '''loss2 = loss_func[1](output_pass2, mask_for_test_unfolded_pass1)'''
        loss2 = loss_func[1](output_pass2, mask_pass2)
        optimizer_pass2.zero_grad()

        loss1.backward(retain_graph=True)
        loss2.backward()

        optimizer_pass1.step()
        optimizer_pass2.step()
        print(str(n_iter) + " iter loss: " + str(loss1.item()) + " " + str(loss2.item()))
        # print(str(n_iter) + " iter loss2: " +)



        # save state dict snapshot
        if n_iter % config.save_checkpoint_iter == 0 \
                and n_iter > init_n_iter:
            misc.save_2_convolution_states("states.pth",
                                           convolution_network_pass1,
                                           convolution_network_pass2,
                                           optimizer_pass1,
                                           optimizer_pass2,
                                           n_iter, config)

        # save state dict snapshot backup
        if config.save_cp_backup_iter \
                and n_iter % config.save_cp_backup_iter == 0 \
                and n_iter > init_n_iter:
            misc.save_2_convolution_states(f"states_{n_iter}.pth",
                                           convolution_network_pass1,
                                           convolution_network_pass2,
                                           optimizer_pass1,
                                           optimizer_pass2,
                                           n_iter, config)


def main():
    args = parser.parse_args()
    config = misc.get_config(args.config)

    # make checkpoint folder if nonexistent
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(os.path.abspath(config.checkpoint_dir))
        os.makedirs(os.path.abspath(f"{config.checkpoint_dir}/images"))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    transforms = [misc.RandomGreyscale(0.5),
                  T.RandomHorizontalFlip(0.5),
                  T.RandomVerticalFlip(0.5)]

    train_dataset = RetinalTrainingDataset(config.dataset_path,
                                           config.mask_folder_path,
                                           img_shape=config.img_shapes,
                                           image_patch_size=config.img_patch_shapes,
                                           image_patch_stride=config.img_stride,
                                           random_crop=config.random_crop,
                                           scan_subdirs=config.scan_subdirs,
                                           transforms=transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True)

    # set random seed
    if config.random_seed != False:
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        import numpy as np
        np.random.seed(config.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available()
                                    and config.use_cuda_if_available else 'cpu')

    input_size_in = config.img_shapes[2]

    trained_convolution_pass1 = RetinalConvolutionNetwork(cnum_in=input_size_in, cnum_out=1)
    trained_convolution_pass1 = trained_convolution_pass1.to(device)
    optim_params_pass1 = trained_convolution_pass1.parameters()
    train_optimizer_pass1 = torch.optim.Adam(
        params=optim_params_pass1,
        lr=config.optimizer_lr,
        weight_decay=config.weight_decay,
        betas=(config.optimizer_beta1_momentum, config.optimizer_beta2))

    trained_convolution_pass2 = RetinalConvolutionNetwork(cnum_in=input_size_in + 1, cnum_out=1)
    trained_convolution_pass2 = trained_convolution_pass2.to(device)
    optim_params_pass2 = trained_convolution_pass2.parameters()
    train_optimizer_pass2 = torch.optim.Adam(
        params=optim_params_pass2,
        lr=config.optimizer_lr,
        weight_decay=config.weight_decay,
        betas=(config.optimizer_beta1_momentum, config.optimizer_beta2))

    if config.gan_loss == 'wbce':
        train_loss = [WeightedBCELoss(50, 100),
                      WeightedBCELoss(30, 100)]  # nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]))
    elif config.gan_loss == 'bce':
        train_loss = [nn.BCEWithLogitsLoss(weight=torch.tensor([10.0])),
                      nn.BCEWithLogitsLoss(weight=torch.tensor([10.0]))]
    elif config.gan_loss == 'avg':
        train_loss = [nn.MSELoss(), nn.MSELoss()]
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    last_n_iter = -1
    if config.model_restore != '':
        state_dicts = torch.load(config.model_restore)
        trained_convolution_pass1.load_state_dict(state_dicts['convolution_pass1'])
        trained_convolution_pass2.load_state_dict(state_dicts['convolution_pass2'])
        if 'adam_opt_pass1' in state_dicts.keys():
            train_optimizer_pass1.load_state_dict(state_dicts['adam_opt_pass1'])
        if 'adam_opt_pass2' in state_dicts.keys():
            train_optimizer_pass2.load_state_dict(state_dicts['adam_opt_pass2'])
        last_n_iter = int(state_dicts['n_iter'])
        print(f"Loaded models from: {config.model_restore}!")

    # start tensorboard logging
    # if config.tb_logging:
    #    from torch.utils.tensorboard import SummaryWriter
    #    writer = SummaryWriter(config.log_dir)

    # display_config = {}
    # display_config.unfolded_size = train_dataset.unfoldedSize()

    training_loop(trained_convolution_pass1,
                  trained_convolution_pass2,
                  train_optimizer_pass1,
                  train_optimizer_pass2,
                  train_loss,
                  train_dataloader,
                  last_n_iter,
                  config)


if __name__ == '__main__':
    main()

"""recombined = torch.zeros_like(output_pass1).to(device)
recombined.copy_(output_pass1)
recombined = torch.clamp_min(recombined, 0.1)
recombined = recombined.permute(1, 0, 2, 3).view(1, mask_unfold_count_h, mask_unfold_count_w,
                                                 image_patch_size[0], image_patch_size[1])
recombined = recombined.permute(0, 1, 3, 2, 4).reshape(1, mask_unfold_count_h * image_patch_size[0],
                                                       mask_unfold_count_w * image_patch_size[1])"""

"""mask_for_test_unfolded_pass2 = mask_pass.unfold(1, image_patch_size[0], image_patch_stride[0])
mask_for_test_unfolded_pass2 = mask_for_test_unfolded_pass2.unfold(2, image_patch_size[1],
                                                                   image_patch_stride[1])
mask_for_test_unfolded_pass2 = mask_for_test_unfolded_pass2.contiguous()
mask_for_test_unfolded_pass2 = mask_for_test_unfolded_pass2.view(1, -1, image_patch_size[0],
                                                                 image_patch_size[1])
mask_for_test_unfolded_pass2 = mask_for_test_unfolded_pass2.permute(1, 0, 2, 3)"""

'''elif config.gan_loss == 'wgbce':
        train_loss = [
            WeightedBCELoss2(50, 100,
                             gauss_params=[config.img_patch_shapes[0],
                                           config.img_patch_shapes[1],
                                           1, 1.6, 20]),  # corner_value, center_value, sigma
            WeightedBCELoss2(30, 100,
                             # blur_use=True,
                             gauss_params=[config.img_patch_shapes[0],
                                           config.img_patch_shapes[1],
                                           1, 1.6, 20])]'''