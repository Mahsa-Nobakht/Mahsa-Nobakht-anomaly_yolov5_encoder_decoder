
import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
import torch
import torch.nn.functional as F


class DAEDataset(Dataset):
    def __init__(self, dataset_path='./dataset/temp', img_type="img", resize=(219, 219),
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], fps=30, video_snippets=32):
        self.dataset_path = dataset_path
        self.img_type = img_type
        # self.type_selector = {"img": "Img", "dimg": "Dimg"}
        self.imgs = self.load_dataset_folder()
        self.resize = resize
        self.mean = mean
        self.std = std
        self.fps = fps
        self.video_snippets = video_snippets

    def __getitem__(self, idx):
        imgs = self.imgs[idx]
        transform_x = T.Compose([T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])
        # if img.endswith('.avi'):
        if imgs.endswith('.avi') or imgs.endswith('.mp4'):
            imgs = self.video_to_frames(imgs, self.fps, self.video_snippets)
            
            # img = frames
            # img = []
            # for frame in imgs:
            #     img = transform_x(frame)
            #     img = self.estimate_background(img)
            #     img.append(img)
            if isinstance(imgs, list):
                # Handle the case where imgs is a list of frames
                processed_frames = [transform_x(frame) for frame in imgs]
                img = torch.stack(processed_frames)  # Stack frames into a single tensor
                img = self.estimate_background(img)
            else:
                img = transform_x(imgs)
                img = self.estimate_background(img)  # Estimate background
               


        else:
            img = cv2.imread(imgs)
            img = transform_x(img)
            # img = self.estimate_background(img)  # Estimate background
            # img = cv2.resize(img, (self.resize[0], self.resize[1]))
            if isinstance(img, list):
                # Handle the case where img is a list of frames
                processed_frames = [cv2.resize(frame, (self.resize[0], self.resize[1])) for frame in img]
                img = torch.stack(processed_frames)  # Stack frames into a single tensor
                img = self.estimate_background(img)
                
            else:
                # img = cv2.resize(img, (self.resize[0], self.resize[1]))
                img = self.estimate_background(img)  # Estimate background
        # Resize images
        # img = F.interpolate(img.unsqueeze(0), size=(self.resize[0], self.resize[1]), mode='bilinear', align_corners=False)
        # img = img.squeeze(0)

        # img = img.permute(1, 2, 0)  # Change the order of dimensions
        # img = F.interpolate(img.unsqueeze(0), size=(self.resize[0], self.resize[1]), mode='bilinear', align_corners=False)
        # img = img.squeeze(0).permute(2, 0, 1)  # Change the order of dimensions back
        # img = img.resize((self.resize[0], self.resize[1]), Image.LANCZOS)
        # img = cv2.resize(img, (self.resize[0], self.resize[1]))
        # Convert RGB image to grayscale
        # img = img.convert('L')
        # img_shape = len(img) #len equls to 1504
        # img = np.array(img)
        # transform_x = T.Compose([T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])
        
        
        # Background estimation using MoG
        
        # img_shape = len(img)
        # Ensure binary_mask is 2D (grayscale) with the same dimensions as the image
        # binary_mask = cv2.resize(binary_mask, (self.resize[0], self.resize[1]))
        # binary_mask = (binary_mask > 0).astype(np.uint8)  # Convert to binary (0 or 1) mask
        # img = img.astype(float)  # Convert to float64

        # Resize the binary_mask to match the dimensions of img
        # binary_mask = cv2.resize(binary_mask, (self.resize[0], self.resize[1])).astype(float)
        # binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]))
        # binary_mask = cv2.resize(binary_mask, (self.resize[0], img.shape[1]))

        # Convert the RGB image to grayscale
        # img = Image.fromarray((img * 255).astype(np.uint8)).convert('L')
        # img_gray = img.copy()
        # img = np.array(img)
        # img = np.repeat(img, 3, axis=0)
        # img = img.astype(float)  # Convert to float64


        # img = Image.fromarray((img * 255).astype(np.uint8))
        
        return img

    def __len__(self):
        return len(self.imgs)

    def load_dataset_folder(self):
        img_paths = []
        # for dataset_path in self.dataset_path:
        vid_dir = os.path.join(self.dataset_path)
        for root, dirs, files in os.walk(vid_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.avi', '.mp4', '.tif')):
                    img_paths.append(os.path.join(root, file))
        return img_paths

        # vid = []
        # vid_dir = os.path.join(self.dataset_path)
        # vid_fpath_list = sorted([os.path.join(vid_dir, f)
        #                          for f in os.listdir(vid_dir)
        #                          if f.endswith(('.jpg', '.png')) or f.endswith('.avi')])
        # vid.extend(vid_fpath_list)
        # return list(vid)

    def video_to_frames(self, video_path, fps, video_snippets):
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        snippet_length = frame_count // video_snippets
        for _ in range(video_snippets):
            for _ in range(snippet_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video position to the beginning
        cap.release()
        return frames

    def estimate_background(self, frames):
        # Background estimation using MoG (Mixture of Gaussians)
        mog = cv2.createBackgroundSubtractorMOG2()
        binary_masks = []  # To store binary masks
        for frame in frames:
            frame_array = np.array(frame, dtype=np.uint8)
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)  # Ensure frame is in BGR format
            # frame_array = cv2.resize(frame_array, (width, height))  # Resize if needed

            mask = mog.apply(frame_array)
            mask = cv2.resize(mask, (self.resize[0], self.resize[1]))
            # Resize the mask to match the image dimensions
            # mask = cv2.resize(mask, (self.resize[0], self.resize[1]))
            # Convert to binary (0 or 1) mask
            # mask = (mask > 0).astype(np.uint8)
            binary_masks.append(mask)

        # Assuming all masks have the same shape, pick one for returning (you can average them if needed)
        return binary_masks

# import os
# from PIL import Image,ImageFilter
# from torch.utils.data import Dataset
# from torchvision import transforms as T
# import numpy as np
# import cv2

# class DAEDataset(Dataset):
#     def __init__(self,dataset_path='./dataset/temp',img_type="img",resize=(219, 219), video_snippets=32,
#                  mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
#         self.dataset_path = dataset_path
#         self.img_type = img_type
#         self.imgs = self.load_dataset_folder()
#         self.resize=resize
#         self.mean=mean
#         self.std=std
#         self.video_snippets = video_snippets 
#         # self.transform = T.Compose([T.Resize((self.resize[0],self.resize[1]), Image.ANTIALIAS),
#         #                        T.ToTensor(),T.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
#         self.transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

#     def __getitem__(self, idx):
#         img = self.imgs[idx]
#         # img = Image.open(img)

#         if img.endswith(('.avi', '.mp4')):
#             # img = self.normalize_image(frames)
#             img = self.video_to_frames(img, video_snippets=32 )
#             img = self.estimate_background(img)
            
#         else:
#             img = Image.open(img)
            
#             img = self.transform(img)
#             # img = self.normalize_image(img)
#             # img = self.MoG.apply(img)
#             img = img.resize((self.resize[0], self.resize[1]))
#             img = self.estimate_background(img)
    
#         return img

#     def __len__(self):
#         return len(self.imgs)

#     def load_dataset_folder(self):
#         # imgs = []
#         # img_dir = os.path.join(self.dataset_path)

#         # img_fpath_list = sorted([os.path.join(img_dir, f)
#         #                         for f in os.listdir(img_dir)
#         #                         if f.endswith(('.jpg','.png', '.avi', '.mp4', '.tif'))])
#         # imgs.extend(img_fpath_list)
#         # return list(imgs)
    
#         img_paths = []
#         # for dataset_path in self.dataset_path:
#         vid_dir = os.path.join(self.dataset_path)
#         for root, dirs, files in os.walk(vid_dir):
#             for file in files:
#                 if file.endswith(('.jpg', '.png', '.avi', '.mp4', '.tif')):
#                     img_paths.append(os.path.join(root, file))
#         return img_paths
    
#     def video_to_frames(self, video_path, video_snippets=32):
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         snippet_length = frame_count // video_snippets
#         for _ in range(video_snippets):
#             for _ in range(snippet_length):
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = Image.fromarray(frame)  # Convert NumPy array to PIL Image
                
#                 frame = self.transform(frame)
#                 frame = np.array(frame)
#                 # frame = cv2.resize(frame, (self.resize[0], self.resize[0]))
#                 # frame = frame.resize((self.resize[0], self.resize[1]), Image.LANCZOS)
#                 frames.append(frame)

#         cap.release()
#         return frames
    
#     def estimate_background(self, frames):
#         # Background estimation using MoG (Mixture of Gaussians)
#         mog = cv2.createBackgroundSubtractorMOG2()
#         binary_masks = []  # To store binary masks
#         for frame in frames:
#             frame_array = np.array(frame)
#             mask = mog.apply(frame_array)
#             # Resize the mask to match the image dimensions
#             mask = cv2.resize(mask, (self.resize[0], self.resize[1]))
#             # Convert to binary (0 or 1) mask
#             # mask = (mask > 0).astype(np.uint8)
#             binary_masks.append(mask)

#         # Assuming all masks have the same shape, pick one for returning (you can average them if needed)
#         return binary_masks
    

