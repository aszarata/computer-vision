from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import data as d
import os

os.makedirs("models", exist_ok=True)
os.makedirs("generated", exist_ok=True)


def custom_training(generator, optimizer_g, discriminator, optimizer_d, data_loader, batch_size, start_epoch,
                    num_epochs, verbose, checkpoint, device, latent_dim, criterion=nn.BCELoss()):
    print("START")

    discriminator.train()
    generator.train()

    fixed_noise = torch.randn(batch_size, latent_dim).to(device)

    # progress bar
    pbar = tqdm(total=verbose * len(data_loader),
                desc=f"Epochs {start_epoch + 1}-{start_epoch + verbose}/{num_epochs + start_epoch}", unit="batch")

    for epoch in range(num_epochs):
        real_epoch = epoch + start_epoch
        # training loop
        for i, (X_batch_real, y_batch_real) in enumerate(data_loader):
            X_batch_real, y_batch_real = X_batch_real.to(device, non_blocking=True), y_batch_real.to(device,
                                                                                                     non_blocking=True)

            """
                Train discriminator.
            """

            ## all real batch
            discriminator.zero_grad()

            y_batch_real -= torch.rand_like(
                y_batch_real) * 0.05  # modify the y_batch by randomizing the labels with noise
            y_pred_real = discriminator(X_batch_real)

            err_d_real = criterion(y_pred_real, y_batch_real)
            err_d_real.backward()

            ## all fake batch
            noise_batch_fake = torch.randn(y_batch_real.shape[0], latent_dim, device=device)
            y_batch_fake = torch.zeros_like(y_batch_real).to(device) + torch.rand_like(y_batch_real) * 0.05  # modify the labels by random noise

            X_batch_fake = generator(noise_batch_fake)

            y_pred_fake = discriminator(X_batch_fake.detach())

            err_d_fake = criterion(y_pred_fake, y_batch_fake)
            err_d_fake.backward()

            # calculate error for discriminator
            err_d = err_d_real + err_d_fake
            # train discriminator
            optimizer_d.step()

            """
                Train generator.
            """
            generator.zero_grad()

            y_pred_gen = discriminator(X_batch_fake)
            y_batch_gen = torch.ones_like(y_batch_real).to(device)

            err_g = criterion(y_pred_gen, y_batch_gen)
            err_g.backward()

            optimizer_g.step()

            """
                Display proggress
            """

            pbar.update(1)
            pbar.set_postfix({"D_loss": err_d.item(), "G_loss": err_g.item()})

        if (real_epoch + 1) % verbose == 0:
            pbar.close()

            gen_test = generator(fixed_noise.detach())
            d.show_tensor_images(gen_test.detach().cpu()[:9], save="generated",
                                 filename=f"checkpoint_epoch_{real_epoch}.png", show=False)
            pbar = tqdm(total=verbose * len(data_loader),
                        desc=f"Epochs {real_epoch + 2}-{real_epoch + 1 + verbose}/{num_epochs + start_epoch}",
                        unit="batch")

        if (real_epoch + 1) % checkpoint == 0:
            save_models(discriminator, optimizer_d, generator, optimizer_g, real_epoch)

        torch.cuda.empty_cache()

    return generator, discriminator


def save_models(model_d, optim_d, model_g, optim_g, epoch, path='models'):
    torch.save({
        'epoch': epoch,
        'model_d_state_dict': model_d.state_dict(),
        'optimizer_d_state_dict': optim_d.state_dict(),
        'model_g_state_dict': model_g.state_dict(),
        'optimizer_g_state_dict': optim_g.state_dict(),
    }, f"{path}/checkpoint_epoch_{epoch}.pth")


def load_models(model_d, optim_d, model_g, optim_g, path):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']

    model_d.load_state_dict(checkpoint['model_d_state_dict'])
    model_g.load_state_dict(checkpoint['model_g_state_dict'])

    optim_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    optim_g.load_state_dict(checkpoint['optimizer_g_state_dict'])

    return epoch
