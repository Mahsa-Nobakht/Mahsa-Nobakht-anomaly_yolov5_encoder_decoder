from torchvision import transforms
import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision import utils
from model.AE_model import ConvAutoencoder  # Import ConvAutoencoder model
from dataset import DAEDataset  # Import DAEDataset
from PIL import Image
from video_data import Video


def train_DAE(args):
    ckpt_path = os.path.join(args.output_path, 'checkpoint')
    sample_path = os.path.join(args.output_path, 'train_samples')
    
    if not os.path.exists(args.output_path):
    # Create the directory if it doesn't exist
        os.makedirs(args.output_path)

    log_path = os.path.join(args.output_path, 'train_log.log')
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines("\n\n====== New Training Loop ======\n\n")
        f.writelines("== Train argument: ==:\n")
        for item in args.__dict__.items():
            f.writelines(str(item) + "\n")
        f.writelines("\n" + "== Train log: ==")

    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    # dataset = DAEDataset(dataset_path=args.dataset_path, img_type=args.img_type)
    dataset = Video(dataset=args.dataset_path)
    # dataset = eval('datasets.get_data')
    # dataset, mask = zip(*data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    class_name = os.path.basename(args.dataset_path)

    # Initialize your model (ConvAutoencoder)
    model = ConvAutoencoder().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.6)

    for epoch in range(args.train_epochs):
        model.train()
        if epoch == 1:
            print("== first epoch done! ==")
        sample_size = 10

        for i, img in enumerate(loader):
            # try:
            # img = Image.open(img)
            # img = transforms.ToPILImage()
            # except FileNotFoundError as e:
                # print(f"Warning: Image not found at path {img}. Skipping...")
                # continue
            # img = img.float()
            # img = torch.tensor(img, dtype=torch.float32)
            if isinstance(img, list):
                # Handle the case where img is a list of frames
                # processed_frames = [torch.tensor(frame).float() for frame in img]
                processed_frames = [frame.clone().detach().float() for frame in img]

                img = torch.stack(processed_frames)  # Stack frames into a single tensor
            else:
                img = img.float()



            # img = torch.tensor(img).float()
            img = img.to(args.device)
            # img = img.clone().detach()
            out, mse = model(img)
            loss = mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            torch.cuda.empty_cache() 
            
            if (epoch + 1) % args.save_gap == 0 and i % (int(len(loader) / 5) + 1) == 0:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.writelines("epoch:{}-{}; loss:{:.3f}; lr:{:.5f}".format(epoch + 1, i + 1, loss.item(), lr) + '\n')

        if (epoch + 1) % args.save_gap == 0:
            model.eval()
            sample = img[:sample_size]
            with torch.no_grad():
                out, _= model(sample)
            
            # Save the reconstructed imagescpu
            utils.save_image(torch.cat([sample, out], 0),
                os.path.join(sample_path, "{}_{}_{}.jpg".format(class_name, args.img_type, epoch + 1)),
                nrow=sample_size,
                normalize=True,
                # range=(-1, 1)
                )

            model.train()
            torch.save(model, os.path.join(ckpt_path, "{}_{}_{}.pt".format(class_name, args.img_type, epoch + 1)))
            print('Save samples and checkpoint, at epoch {}'.format(epoch + 1))
        scheduler.step()
    print("{} training process done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_type', type=str, default='img', choices=['img', 'dimg'], help='input img or dimg')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--save_gap', type=int, default=10, help="save by this frequency")
    parser.add_argument('--dataset_path', type=str, default='dataset/avenue/training_videos', help='dataset path')
    parser.add_argument('--output_path', type=str, default='output', help='path to save log and ckpt')
    parser.add_argument('--device', type=str, default='cuda', help='device number')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    args = parser.parse_args()
    args.img_type = 'img'
    train_DAE(args)
    # args.img_type = 'dimg'
    # train_DAE(args)
