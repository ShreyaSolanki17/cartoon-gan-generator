import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.tensorboard import SummaryWriter

from model import ATTR_MAXES, FIXED_ATTRS, CartoonGenerator, AttributePredictor

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE_G = 5e-4
LEARNING_RATE_D = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TENSOR_DIR = "cartoonset100k_tensors"
MODEL_SAVE_DIR = "models"
LOG_DIR = "runs"

# Setup directories
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Dataset class
class CartoonDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.tensor_files = [f for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        self.tensor_files.sort()
        print(f"Found {len(self.tensor_files)} samples in {tensor_dir}")

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.tensor_dir, self.tensor_files[idx]))
        return data["img"], data["outline"], data["attrs"]

# Model definitions (unchanged)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1)
        )

    def forward(self, x):
        return self.main(x)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(DEVICE)
        self.layers = vgg
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return nn.MSELoss()(self.layers(x), self.layers(y))

# Training utilities (gradient_penalty and validate functions unchanged)
def gradient_penalty(disc, real_imgs, fake_imgs):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=DEVICE)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolates = disc(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def validate(gen, attr_pred, disc, dataloader, color_criterion, outline_criterion, attr_criterion, perceptual_criterion, adv_criterion, device, writer, epoch):
    gen.eval()
    attr_pred.eval()
    disc.eval()
    total_color_loss, total_outline_loss, total_attr_loss, total_adv_loss, total_hair_loss = 0, 0, 0, 0, 0
    total_face_shape_acc, total_hair_acc, total_glasses_acc, batches = 0, 0, 0, 0

    with torch.no_grad():
        for real_imgs, real_outlines, attrs in dataloader:
            real_imgs = real_imgs.to(device, non_blocking=True)
            real_outlines = real_outlines.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            batch_size = real_imgs.size(0)

            fake_imgs, fake_outlines = gen(attrs)
            color_loss = color_criterion(fake_imgs, real_imgs)
            face_outline_loss = outline_criterion(fake_outlines[:, 0:1], real_outlines)
            hair_beard_loss = outline_criterion(fake_outlines[:, 1:2], real_outlines)
            outline_loss = 0.6 * face_outline_loss + 0.4 * hair_beard_loss
            hair_mask = fake_outlines[:, 1:2].detach()
            hair_texture_loss = color_criterion(fake_imgs * hair_mask, real_imgs * hair_mask)
            perceptual_loss = perceptual_criterion(fake_imgs, real_imgs)
            adv_loss = adv_criterion(disc(fake_imgs), torch.full((batch_size, 1), 0.8, device=device))
            attr_preds = attr_pred(fake_imgs)
            attr_loss = sum(attr_criterion(pred, attrs[:, i]) for i, pred in enumerate(attr_preds)) / 18

            face_shape_acc = (torch.argmax(attr_preds[7], dim=1) == attrs[:, 7]).float().mean()
            hair_acc = (torch.argmax(attr_preds[9], dim=1) == attrs[:, 9]).float().mean()
            glasses_acc = (torch.argmax(attr_preds[10], dim=1) == attrs[:, 10]).float().mean()

            total_color_loss += color_loss.item()
            total_outline_loss += outline_loss.item()
            total_attr_loss += attr_loss.item()
            total_adv_loss += adv_loss.item()
            total_hair_loss += hair_texture_loss.item()
            total_face_shape_acc += face_shape_acc.item()
            total_hair_acc += hair_acc.item()
            total_glasses_acc += glasses_acc.item()
            batches += 1

    avg_color_loss = total_color_loss / batches
    avg_outline_loss = total_outline_loss / batches
    avg_attr_loss = total_attr_loss / batches
    avg_adv_loss = total_adv_loss / batches
    avg_hair_loss = total_hair_loss / batches
    avg_face_shape_acc = total_face_shape_acc / batches
    avg_hair_acc = total_hair_acc / batches
    avg_glasses_acc = total_glasses_acc / batches

    writer.add_scalar("Val/Loss/Color", avg_color_loss, epoch + 1)
    writer.add_scalar("Val/Loss/Outline", avg_outline_loss, epoch + 1)
    writer.add_scalar("Val/Loss/HairTexture", avg_hair_loss, epoch + 1)
    writer.add_scalar("Val/Loss/Attribute", avg_attr_loss, epoch + 1)
    writer.add_scalar("Val/Loss/Adversarial", avg_adv_loss, epoch + 1)
    writer.add_scalar("Val/Acc/FaceShape", avg_face_shape_acc, epoch + 1)
    writer.add_scalar("Val/Acc/Hair", avg_hair_acc, epoch + 1)
    writer.add_scalar("Val/Acc/Glasses", avg_glasses_acc, epoch + 1)

    sample_attrs = torch.tensor(FIXED_ATTRS, dtype=torch.int64, device=device).unsqueeze(0)
    fake_imgs, fake_outlines = gen(sample_attrs)
    writer.add_image("Val/Generated", fake_imgs.squeeze(0) * 0.5 + 0.5, epoch + 1)
    writer.add_image("Val/Face Outline", fake_outlines[:, 0:1].squeeze(0) * 0.5 + 0.5, epoch + 1)
    writer.add_image("Val/HairBeard Outline", fake_outlines[:, 1:2].squeeze(0) * 0.5 + 0.5, epoch + 1)

    gen.train()
    attr_pred.train()
    disc.train()

# Main training loop
if __name__ == '__main__':
    train_dirs = [os.path.join(TENSOR_DIR, str(i)) for i in range(8)]
    val_dirs = [os.path.join(TENSOR_DIR, str(i)) for i in [8, 9]]
    
    val_datasets = [CartoonDataset(val_dir) for val_dir in val_dirs]
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    gen = CartoonGenerator().to(DEVICE)
    attr_pred = AttributePredictor().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    opt_g = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
    opt_attr = optim.Adam(attr_pred.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    scheduler_g = optim.lr_scheduler.StepLR(opt_g, step_size=50, gamma=0.7)
    scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=50, gamma=0.7)

    color_criterion = nn.MSELoss()
    outline_criterion = nn.MSELoss()
    attr_criterion = nn.CrossEntropyLoss()
    perceptual_criterion = PerceptualLoss()
    adv_criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(LOG_DIR)

    for epoch in range(NUM_EPOCHS):
        train_dir = train_dirs[epoch % 8]
        train_dataset = CartoonDataset(train_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        for i, (real_imgs, real_outlines, attrs) in enumerate(train_dataloader):
            real_imgs = real_imgs.to(DEVICE, non_blocking=True)
            real_outlines = real_outlines.to(DEVICE, non_blocking=True)
            attrs = attrs.to(DEVICE, non_blocking=True)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            opt_d.zero_grad()
            fake_imgs, fake_outlines = gen(attrs)
            real_labels = torch.full((batch_size, 1), 0.8, device=DEVICE)
            fake_labels = torch.full((batch_size, 1), 0.2, device=DEVICE)
            d_real = disc(real_imgs)
            d_fake = disc(fake_imgs.detach())
            d_loss = (adv_criterion(d_real, real_labels) + adv_criterion(d_fake, fake_labels)) / 2
            gp = gradient_penalty(disc, real_imgs, fake_imgs.detach())
            d_loss = d_loss + 0.1 * gp
            d_loss.backward()
            opt_d.step()

            # Train Generator + Attr Predictor
            opt_g.zero_grad()
            opt_attr.zero_grad()
            fake_imgs, fake_outlines = gen(attrs)
            color_loss = color_criterion(fake_imgs, real_imgs)
            face_outline_loss = outline_criterion(fake_outlines[:, 0:1], real_outlines)
            hair_beard_loss = outline_criterion(fake_outlines[:, 1:2], real_outlines)
            outline_loss = 0.6 * face_outline_loss + 0.4 * hair_beard_loss
            hair_mask = fake_outlines[:, 1:2].detach()
            hair_texture_loss = color_criterion(fake_imgs * hair_mask, real_imgs * hair_mask)
            perceptual_loss = perceptual_criterion(fake_imgs, real_imgs)
            adv_loss = adv_criterion(disc(fake_imgs), real_labels)
            attr_preds = attr_pred(fake_imgs)
            attr_loss = sum(attr_criterion(pred, attrs[:, i]) for i, pred in enumerate(attr_preds)) / 18

            if epoch < 50:
                total_loss = 0.35 * outline_loss + 0.25 * attr_loss + 0.15 * color_loss + 0.15 * perceptual_loss + 0.1 * adv_loss + 0.05 * hair_texture_loss
            elif epoch < 100:
                total_loss = 0.3 * outline_loss + 0.25 * attr_loss + 0.15 * color_loss + 0.15 * perceptual_loss + 0.1 * adv_loss + 0.05 * hair_texture_loss
            else:
                total_loss = 0.25 * outline_loss + 0.2 * attr_loss + 0.2 * color_loss + 0.2 * perceptual_loss + 0.1 * adv_loss + 0.05 * hair_texture_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(attr_pred.parameters(), max_norm=1.0)
            opt_g.step()
            opt_attr.step()

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % 4 == 0:
            validate(gen, attr_pred, disc, val_dataloader, color_criterion, outline_criterion, 
                     attr_criterion, perceptual_criterion, adv_criterion, DEVICE, writer, epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), os.path.join(MODEL_SAVE_DIR, f"gen_epoch_{epoch+1}.pth"))

    torch.save(gen.state_dict(), os.path.join(MODEL_SAVE_DIR, "gen_final.pth"))
    writer.close()