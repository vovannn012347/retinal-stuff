import os
import time
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as t_tf
import torchvision.transforms.functional as t_tf_fn

import model.losses_inpaint as gan_losses
import utils.misc_inpaint as misc
from utils.edge_detection import sobel_edge_detection_batch, prewitt_edge_detection_batch
from utils.retinaldata import RetinalFCNNMaskDataset

from utils.misc_retina import run_retina_cnn_2, magic_wand_mask_selection_batch
from model.networks_inpaint import Generator, Discriminator
from model.retina_network import RetinalConvolutionNetwork
from model.retina_network import WeightedBCELoss


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default="configs/train_retina_inpaint.yaml", help="Path to yaml config file")


class UniformMaskedRadialSpreader(nn.Module):
    def __init__(self, radius, wand_tolerance, gauss_sigma=30, device='cpu'):
        super(UniformMaskedRadialSpreader, self).__init__()

        self.training = False

        size = 2 * radius + 1
        center = radius
        tensor = torch.zeros((size, size), dtype=torch.float32)

        center2 = center ** 2
        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= center2:
                    tensor[i, j] = 1.0

        channels = 3
        self.kernel_1ch = tensor.unsqueeze(0).unsqueeze(0)
        self.kernel_3ch = self.kernel_1ch.expand(channels, 1, size, size)

        self.conv_1ch = torch.nn.Conv2d(1, 1, kernel_size=size, groups=1, bias=False, padding=size // 2)
        self.conv_1ch.weight.data = self.kernel_1ch
        self.conv_1ch.weight.requires_grad = False

        self.conv_3ch = torch.nn.Conv2d(channels, channels,
                                        kernel_size=size, groups=channels, bias=False, padding=size // 2)
        self.conv_3ch.weight.data = self.kernel_3ch
        self.conv_3ch.weight.requires_grad = False

        self.gaussian_conv = misc.get_gaussian_conv2d(device, channels, size, gauss_sigma)

        self.eps = 1e-6
        self.tolerance = wand_tolerance

    # batch images are expected to be [-1.0 ... 1.0]
    # masks are expected to be 0.0/1.0
    def forward(self, batch_images: torch.Tensor, batch_mask: torch.Tensor):

        batch_dark_area_mask = torch.zeros_like(batch_mask)

        start_x = 0
        start_y = 0
        for mask_index in range(batch_images.shape[0]):
            mask = batch_dark_area_mask[mask_index, :, :, :]
            image = batch_images[mask_index, :, :, :]
            stack = [(start_x, start_y),
                     (image.shape[1] - 1 - start_x, start_y),
                     (image.shape[1] - 1 - start_x, image.shape[2] - 1 - start_y),
                     (start_x, image.shape[2] - 1 - start_y)]
            start_color = image[:, start_x, start_y].clone()
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[2]:
                    continue
                if mask[0, x, y] > 0:
                    continue

                current_color = image[:, x, y]
                if torch.sqrt(torch.sum((current_color - start_color) ** 2)) > self.tolerance:
                    continue

                mask[0, x, y] = 1.0  # Mark the position as visited

                # Add neighboring positions to the stack
                stack.append((x + 1, y))
                stack.append((x - 1, y))
                stack.append((x, y + 1))
                stack.append((x, y - 1))

        batch_mask_inverted = 1 - batch_mask
        batch_images_normalized = (batch_images + 1) / 2
        batch_blur_mask = (batch_mask_inverted - batch_dark_area_mask).clamp(min=0.0)
        batch_image_mask = (1 - batch_dark_area_mask)

        batch_images_pre_blur = batch_images_normalized * batch_blur_mask

        batch_mask_blurred = self.conv_1ch(batch_mask_inverted)
        batch_image_blurred = self.conv_3ch(batch_images_pre_blur)

        batch_mask_missed = (batch_mask_blurred < self.eps).float()  # important to care about 0 division

        batch_mask_blurred_divisor = batch_mask_blurred * batch_mask
        batch_image_blurred_divisible = batch_image_blurred * (batch_mask - batch_mask_missed)

        # contains only blurred result excluding unmasked areas not toucnhed by blur
        batch_image_blurred_result = ((batch_image_blurred_divisible /
                                      (batch_mask_blurred_divisor + batch_mask_inverted + batch_mask_missed))
                                      * batch_image_mask)

        batch_images_nonblurred_result = batch_images_normalized * (batch_mask_inverted + batch_mask_missed)

        batch_images_result = batch_image_blurred_result + batch_images_nonblurred_result
        # (((batch_image_blurred_result + batch_images_nonblurred_result) * 2 - 1)
        # .clamp(min=-1.0, max=1.0)))

        # batch_gauss_result = self.gaussian_conv(batch_images_result)

        # batch_smoothed_images_result = batch_gauss_result * batch_mask + batch_images_result * batch_mask_inverted

        # batch_smoothed_images_result = (batch_smoothed_images_result * 2 - 1).clamp(min=-1.0, max=1.0)

        # viz_images = [batch_images_normalized,
        #               batch_images_pre_blur,
        #               ((batch_mask_inverted - batch_dark_area_mask).clamp(min=0.0)).repeat(1, 3, 1, 1),
        #               batch_mask_inverted.repeat(1, 3, 1, 1),
        #               batch_dark_area_mask.repeat(1, 3, 1, 1)]
        #
        #
        # img_grid = torch.cat(torch.cat(viz_images, dim=3).unbind(dim=0), dim=1).unsqueeze(0)
        # img_out = TF.to_pil_image(img_grid.squeeze().cpu(), mode="RGB")
        # global save_dir
        # img_out.save(f"{save_dir}/images/iter_test.png")

        batch_images_result = (batch_images_result * 2 - 1).clamp(min=-1.0, max=1.0)

        return batch_images_result


def training_loop(retina_pass1,     # retina pass
                  generator,        # generator network
                  discriminator,    # discriminator network
                  g_optimizer,      # generator optimizer
                  d_optimizer,      # discriminator optimizer
                  gan_loss_g,        # generator gan loss function
                  gan_loss_d,        # discriminator gan loss function
                  bin_loss,          # binary mask 0..1 loss function
                  train_dataloader,  # training dataloader
                  last_n_iter,      # last iteration
                  config            # Config object
                  ):

    device = torch.device('cpu')

    losses = {}

    generator.train()
    discriminator.train()

    # initialize dict for logging
    losses_log = {'d_loss':   [],
                  'g_loss':   [],
                  'ae_loss':  [],
                  'ae_loss1': [],
                  'ae_loss2': []
                  }

    # if config.retina_loss:
    #     losses_log['g_loss_ret'] = []

    # if config.edge_loss:
    #     losses_log['edg_loss'] = []

    # training loop
    init_n_iter = last_n_iter + 1
    train_iter = iter(train_dataloader)
    time0 = time.time()
    # image_target_size = config.img_shapes # image is resized in data loader
    img_stride = config.img_stride
    img_patch_shape = config.img_patch_shapes

    to_pil_grey_transform = t_tf.ToPILImage(mode='L')
    to_pil_transform = t_tf.ToPILImage(mode='RGB')

    target_color = 0
    tolerance = 0.23

    zero_values = torch.zeros(
        config.batch_size,
        1,
        config.img_shapes[0],
        config.img_shapes[1])

    gaussian_blur_convo = UniformMaskedRadialSpreader(19, 0.25)

    # gaussian_convo = get_gaussian_conv2d(device, config.img_shapes[2], 19, 32)
    # gaussian_mask_convo = get_gaussian_conv2d(device, 1, 19, 32)

    # learning is done in singular items
    for n_iter in range(init_n_iter, config.max_iters):
        # load batch of raw data
        try:
            batch_real_images, batch_retina_mask = next(train_iter)
        except:
            train_iter = iter(train_dataloader)
            batch_real_images, batch_retina_mask = next(train_iter)

        print(f"@iter: {n_iter}")
        batch_real_images = batch_real_images.to(device, non_blocking=True)

        if config.do_innard_testing_image_shots:
            to_pil_grey_transform(batch_retina_mask[0]).save(f"{config.checkpoint_dir}/images/iter_mask_{n_iter}.png")
            to_pil_transform((batch_real_images[0] + 1)/2).save(f"{config.checkpoint_dir}/images/iter_img_{n_iter}.png")

        # create mask
        # randomly puts squares, lines, ellipses and etceteral stuff to make irregular mask
        bbox = misc.random_bbox(config)
        regular_mask = misc.bbox2mask(config, bbox).to(device)
        irregular_mask = misc.brush_stroke_mask(config).to(device)

        # here we do need simple binary selection mask
        cleaned_batch_retina_mask = magic_wand_mask_selection_batch(batch_retina_mask).to(torch.float32)
        cleaned_batch_retina_mask = misc.dilate_mask(cleaned_batch_retina_mask, 3)
            #misc.dilate_mask(((abs(batch_retina_mask - target_color) > tolerance).to(torch.float32)), 3))

        inverted_batch_retina_mask = 1 - cleaned_batch_retina_mask

        # expands masks to all images
        mask = torch.logical_or(irregular_mask, regular_mask)
        mask_expanded = mask.expand(batch_retina_mask.size(0), -1, -1, -1)
        mask = torch.logical_or(mask_expanded, cleaned_batch_retina_mask)
        mask = mask.to(torch.float32).to(device)

        if config.do_innard_testing_image_shots:
            to_pil_grey_transform(mask[0]).save(f"{config.checkpoint_dir}/images/iter_mask_gen_{n_iter}.png")

        # prepare input for generator
        batch_incomplete = batch_real_images*(1.-mask)

        if config.do_innard_testing_image_shots:
            (to_pil_transform((batch_incomplete[0] + 1)/2)
             .save(f"{config.checkpoint_dir}/images/iter_img_icomplete_{n_iter}.png"))

        ones_x = torch.ones_like(batch_incomplete)[:, 0:1].to(device)
        x = torch.cat(tensors=[batch_incomplete, ones_x, ones_x*mask], dim=1)

        # generate inpainted images
        rough_fill, fine_fill = generator(x, mask)  # x1=rough-painted x2=fine-painted
        batch_predicted = fine_fill

        # apply mask and complete image
        batch_complete = batch_real_images*(1.-mask) + batch_predicted * mask

        if config.do_innard_testing_image_shots:
            to_pil_transform((batch_complete[0] + 1) / 2).save(
                f"{config.checkpoint_dir}/images/iter_img_complete_{n_iter}.png")

        batch_filled_image_retina_mask = torch.zeros(
            batch_complete.size(0),
            1,
            batch_complete.size(2),
            batch_complete.size(3))

        # re-run retina cnn to detect whether some blood vessels still present
        # if config.retina_loss:
        #     for i in range(batch_filled_image_retina_mask.size(0)):
        #         batch_filled_image_retina_mask[i] = run_retina_cnn_2(
        #             batch_complete[i],
        #             retina_pass1,
        #             img_patch_shape,
        #             img_stride,
        #             device)
        #
        #     batch_filled_image_retina_mask = (
        #             batch_filled_image_retina_mask *
        #             magic_wand_mask_selection_batch(batch_filled_image_retina_mask).to(torch.float32))

        if config.do_innard_testing_image_shots:
            (to_pil_grey_transform(batch_filled_image_retina_mask[0])
             .save(f"{config.checkpoint_dir}/images/iter_retina_mask_detected_{n_iter}.png"))

        # discriminator is normally trained on base image and on filled images
        # however we are using transfer learning where we want discriminator
        # to transfer infill type from freeform mask to retina mask when doing infill
        # D training steps:
        batch_real_with_infill_likeness = gaussian_blur_convo(batch_real_images.detach(), cleaned_batch_retina_mask)


        # batch_real_mask = torch.cat((batch_real_images, mask), dim=1)
        # previous value above is modified to have infill likeness
        batch_real_mask = torch.cat((batch_real_with_infill_likeness.detach(), mask), dim=1)

        batch_filled_mask = torch.cat((batch_complete.detach(), mask), dim=1)
        batch_real_and_filled = torch.cat((batch_real_mask, batch_filled_mask))

        d_real_gen = discriminator(batch_real_and_filled)
        d_real, d_gen = torch.split(d_real_gen, config.batch_size)

        d_loss = gan_loss_d(d_real, d_gen)
        losses['d_loss'] = d_loss

        # update D parameters
        d_optimizer.zero_grad()
        losses['d_loss'].backward()
        d_optimizer.step()

        # G training steps:
        # rough loss calc
        losses['ae_loss1'] = (config.l1_loss_alpha *
                              torch.mean(inverted_batch_retina_mask * torch.abs(batch_real_images - rough_fill)))

        # fine loss calc
        losses['ae_loss2'] = (config.l1_loss_alpha *
                              torch.mean(inverted_batch_retina_mask * torch.abs(batch_real_images - fine_fill)))

        losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

        batch_gen = batch_predicted
        batch_gen = torch.cat((batch_gen, mask), dim=1)

        d_gen = discriminator(batch_gen)

        losses['g_loss'] = config.gan_loss_alpha * gan_loss_g(d_gen)

        batch_edge_mask_real = None
        batch_edge_mask_filled = None
        batch_edge_mask_loss = None

        # if config.edge_loss:
        #     # if config.edge_function == 'canny':
        #     #     batch_edge_mask_real = canny_edge_detection_batch(batch_real_images)
        #     #     batch_edge_mask_filled = canny_edge_detection_batch(batch_complete)
        #     # el
        #     if config.edge_function == 'sobel':
        #         batch_edge_mask_real = sobel_edge_detection_batch(batch_real_with_infill_likeness.detach())
        #         batch_edge_mask_filled = sobel_edge_detection_batch(batch_complete)
        #     elif config.edge_function == 'prewitt':
        #         batch_edge_mask_real = prewitt_edge_detection_batch(batch_real_with_infill_likeness.detach())
        #         batch_edge_mask_filled = prewitt_edge_detection_batch(batch_complete)
        #
        #     if batch_edge_mask_real is not None:
        #         losses["edg_loss"] = torch.mean((batch_edge_mask_real - batch_edge_mask_filled) ** 2)
        #         batch_edge_mask_loss = torch.abs(batch_edge_mask_filled - batch_edge_mask_real)
        #
        #     if config.do_innard_testing_image_shots:
        #         to_pil_transform(batch_edge_mask_loss[0]).save(
        #             f"{config.checkpoint_dir}/images/iter_img_edge_loss_{n_iter}.png")
        #         to_pil_transform(batch_edge_mask_real[0]).save(
        #             f"{config.checkpoint_dir}/images/iter_img_edge_real_{n_iter}.png")
        #         to_pil_transform(batch_edge_mask_filled[0]).save(
        #             f"{config.checkpoint_dir}/images/iter_img_edge_filled_{n_iter}.png")

        # if config.retina_loss is True:
        #     losses['g_loss_ret'] = (config.bin_retina_loss_alpha *
        #                             bin_loss(batch_filled_image_retina_mask *
        #                                      cleaned_batch_retina_mask, zero_values))
        #     losses['g_loss'] += losses['g_loss_ret']

        if config.ae_loss:
            losses['g_loss'] += losses['ae_loss']

        # if config.edge_loss:
        #     losses['g_loss'] += losses['edg_loss'] * config.edge_loss_alpha

        # update G parameters
        g_optimizer.zero_grad()
        losses['g_loss'].backward()
        g_optimizer.step()

        # LOGGING
        for k in losses_log.keys():
            losses_log[k].append(losses[k].item())

        # (tensorboard) logging
        if n_iter % config.print_iter == 0:
            # measure iterations/second
            dt = time.time() - time0
            print(f"@iter: {n_iter}: {(config.print_iter/dt):.4f} it/s")
            time0 = time.time()

            # write loss terms to console
            # and tensorboard
            for k, loss_log in losses_log.items():
                loss_log_mean = sum(loss_log)/len(loss_log)
                print(f"{k}: {loss_log_mean:.4f}")
                # if config.tb_logging:
                #    writer.add_scalar(
                #        f"losses/{k}", loss_log_mean, global_step=n_iter)
                losses_log[k].clear()

        debug_save = True
        # save example image grids to disk
        if (config.save_imgs_to_disc_iter
                and n_iter % config.save_imgs_to_disc_iter == 0) or debug_save:
            viz_images = []

            # if config.edge_loss:
            #     viz_images = [misc.pt_to_image(batch_real_images),
            #                   misc.pt_to_image(batch_complete),
            #                   cleaned_batch_retina_mask.repeat(1, 3, 1, 1),
            #                   batch_filled_image_retina_mask.repeat(1, 3, 1, 1),
            #                   batch_edge_mask_real.repeat(1, 3, 1, 1),
            #                   batch_edge_mask_filled.repeat(1, 3, 1, 1),
            #                   batch_edge_mask_loss.repeat(1, 3, 1, 1)]
            # elif config.retina_loss:
            #     viz_images = [misc.pt_to_image(batch_real_images),
            #                   misc.pt_to_image(batch_complete),
            #                   cleaned_batch_retina_mask.repeat(1, 3, 1, 1),
            #                   batch_filled_image_retina_mask.repeat(1, 3, 1, 1)]
            # else:
            viz_images = [misc.pt_to_image(batch_real_images),
                          misc.pt_to_image(batch_complete),
                          misc.pt_to_image(batch_real_with_infill_likeness),
                          cleaned_batch_retina_mask.repeat(1, 3, 1, 1)]

            img_grid = torch.cat(torch.cat(viz_images, dim=3).unbind(dim=0), dim=1).unsqueeze(0)
            img_out = t_tf_fn.to_pil_image(img_grid.squeeze().cpu(), mode="RGB")
            img_out.save(f"{config.checkpoint_dir}/images/iter_{n_iter}.png")
            print("Saved image examples!")

            """img_grids = [tv.utils.make_grid(images[:config.viz_max_out], nrow=2)
                                            for images in viz_images]
            tv.utils.save_image(img_grids,f"{config.checkpoint_dir}/images/iter_{n_iter}.png", nrow=2)"""

        # save state dict snapshot
        if n_iter % config.save_checkpoint_iter == 0 and n_iter > init_n_iter:
            misc.save_states("states_gen.pth",
                             generator, discriminator,
                             g_optimizer, d_optimizer,
                             n_iter, config)
            # return

        if n_iter % config.save_testing_cp_iter == 0 and n_iter > init_n_iter:
            misc.save_testing_model_states("states_gen_test.pth", generator, retina_pass1, config)

        # save state dict snapshot backup
        if config.save_cp_backup_iter and n_iter % config.save_cp_backup_iter == 0 and n_iter > init_n_iter:
            misc.save_states(f"states_gen_{n_iter}.pth",
                             generator, discriminator,
                             g_optimizer, d_optimizer,
                             n_iter, config)


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
    
    # construct networks
    input_size_in = config.img_shapes[2]
    # retinal pass methods
    trained_convolution_pass = RetinalConvolutionNetwork(cnum_in=input_size_in, cnum_out=1)

    # filler generator
    generator = Generator(cnum_in=input_size_in+2, cnum_out=input_size_in, cnum=48, return_flow=False)
    # discriminator after testing
    discriminator = Discriminator(cnum_in=input_size_in+1, cnum=64)

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    trained_convolution_pass = trained_convolution_pass.to(device)

    # optimizers
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config.g_lr, betas=(config.g_beta1, config.g_beta2))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config.d_lr, betas=(config.d_beta1, config.d_beta2))

    # losses
    if config.gan_loss == 'hinge':
        gan_loss_d, gan_loss_g = gan_losses.hinge_loss_d, gan_losses.hinge_loss_g
    elif config.gan_loss == 'ls':
        gan_loss_d, gan_loss_g = gan_losses.ls_loss_d, gan_losses.ls_loss_g
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    if config.bin_loss == 'wbce':
        bin_loss = WeightedBCELoss(1, 10)
    elif config.bin_loss == 'bce':
        bin_loss = nn.BCEWithLogitsLoss(weight=torch.tensor([1.0]))
    elif config.bin_loss == 'avg':
        bin_loss = nn.MSELoss()
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    if config.retina_model_restore != '':
        retina_train_state_dicts = torch.load(config.retina_model_restore)
        trained_convolution_pass.load_state_dict(retina_train_state_dicts['convolution_pass1'])
        print(f"Loaded models from: {config.retina_model_restore}!")

    # resume from existing checkpoint
    last_n_iter = -1
    if config.generator_model_restore != '':
        generator_train_state_dicts = torch.load(config.generator_model_restore, map_location=device)

        if 'G' in generator_train_state_dicts.keys():
            generator.load_state_dict(generator_train_state_dicts['G'])
        if 'D' in generator_train_state_dicts.keys():
            discriminator.load_state_dict(generator_train_state_dicts['D'])
        if 'G_optim' in generator_train_state_dicts.keys():
            g_optimizer.load_state_dict(generator_train_state_dicts['G_optim'])
        if 'D_optim' in generator_train_state_dicts.keys():
            d_optimizer.load_state_dict(generator_train_state_dicts['D_optim'])
        if 'gen_iter' in generator_train_state_dicts.keys():
            last_n_iter = generator_train_state_dicts['gen_iter']
        print(f"Loaded models from: {config.generator_model_restore}!")

    # transforms stuff
    # transforms = [misc.RandomGreyscale(0.5), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)] \
    transforms = [misc.RandomGreyscale(0.25), t_tf.RandomHorizontalFlip(0.5), t_tf.RandomVerticalFlip(0.5)] \
        if config.random_transforms else None

    # dataloading
    train_dataset = RetinalFCNNMaskDataset(config.dataset_path,
                                           trained_convolution_pass,
                                           device=device,
                                           img_shape=config.img_shapes,
                                           image_patch_size=config.img_patch_shapes,
                                           image_patch_stride=config.img_stride,
                                           random_crop=config.random_crop,
                                           # provide_greyscale=config.edge_loss,
                                           scan_subdirs=config.scan_subdirs,
                                           transforms=transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,

                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True)

    # start tensorboard logging
    # writer = None
    # if config.tb_logging:
    #    from torch.utils.tensorboard import SummaryWriter
    #    writer = SummaryWriter(config.log_dir)

    # start training
    training_loop(trained_convolution_pass,
                  generator,
                  discriminator,
                  g_optimizer,
                  d_optimizer,
                  gan_loss_g,
                  gan_loss_d,
                  bin_loss,
                  train_dataloader,
                  last_n_iter,
                  # writer,
                  config)


if __name__ == '__main__':
    main()
