import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch as nn
import torch.optim as optim
from model import UNET
from utils import load_checkpoint, save_checkpoint, check_accuracy, get_loaders, save_predictions_as_imgs

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Carvana/train"
VAL_IMG_DIR = "Carvana/valid"
TRAIN_MASK_DIR = "Carvana/train_masks"
VAL_MASK_DIR = "Carvana/valid_masks"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    print("Starting")
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        predictions = model(data)
        loss = nn.binary_cross_entropy_with_logits(predictions, targets)
        loss = loss.sum()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.binary_cross_entropy_with_logits
    optimzer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):

        train_fn(train_loader, model, optimzer, loss_fn, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimzer.state_dict(),
        }
        save_checkpoint(checkpoint)

        #Check accuracy 
        check_accuracy(val_loader, model)

        #Print samples in folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/"
        )

if __name__ == "__main__":
    main()
