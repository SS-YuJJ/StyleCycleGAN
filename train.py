"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os, time

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = opt.total_iter_from                # the total number of training iterations

    inNout_log_path = os.path.join(opt.checkpoints_dir, opt.name, "input_output_log.txt")
    now = time.strftime("%c")
    with open(inNout_log_path, 'a') as f:
        f.write(f"================ New Run Input Output Info {now} ================\n")

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        if total_iters > opt.total_iter_max:
            print("*********** Reached maximum total iter ***********")
            break

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            
            update_D = True if (i+1) % opt.GD_update_ratio == 0 else False
            model.optimize_parameters(update_D)   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()

            # ============== log input & output infos ===============
            wandb.log({'real_A_min': model.real_A.data.min(), 'real_A_max': model.real_A.data.max(),'real_A_mean': model.real_A.data.mean(),'real_A_std': model.real_A.data.std()})
            wandb.log({'real_B_min': model.real_B.data.min(), 'real_B_max': model.real_B.data.max(),'real_B_mean': model.real_B.data.mean(),'real_B_std': model.real_B.data.std()})
            wandb.log({'fake_A_min': model.fake_A.data.min(), 'fake_A_max': model.fake_A.data.max(),'fake_A_mean': model.fake_A.data.mean(),'fake_A_std': model.fake_A.data.std()})
            wandb.log({'fake_B_min': model.fake_B.data.min(), 'fake_B_max': model.fake_B.data.max(),'fake_B_mean': model.fake_B.data.mean(),'fake_B_std': model.fake_B.data.std()})
            wandb.log({'rec_A_min': model.rec_A.data.min(), 'rec_A_max': model.rec_A.data.max(),'rec_A_mean': model.rec_A.data.mean(),'rec_A_std': model.rec_A.data.std()})
            wandb.log({'rec_B_min': model.rec_B.data.min(), 'rec_B_max': model.rec_B.data.max(),'rec_B_mean': model.rec_B.data.mean(),'rec_B_std': model.rec_B.data.std()})
            # =======================================================

            if opt.use_wandb:           # plot all losses
                for k, v in losses.items():
                    if ('D' not in k) or ('D' in k and update_D):
                        wandb.log({f'{k}': v})

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                print(f"----- Total iters = {total_iters} -----")
                if opt.use_visdom and opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters == opt.total_iter_max or ((total_iters % opt.save_latest_freq == 0) and (total_iters != 0)):
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

            if total_iters > opt.total_iter_max:
                print("*********** Reached maximum total iter! ***********")
                break


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
