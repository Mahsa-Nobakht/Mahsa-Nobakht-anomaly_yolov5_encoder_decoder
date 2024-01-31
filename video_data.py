import os
import natsort
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2  # Import OpenCV
import numpy as np



# def make_power_2(n, base=32.0):
#     return int(round(n / base) * base)


def get_transform(size, method=Image.BICUBIC, normalize=True, toTensor=True):
    # w, h = size
    # new_size = [make_power_2(w), make_power_2(h)]

    # transform_list = [transforms.Resize(new_size, method)]
    # transform_list = [transforms.Resize(size, method)]
    transform_list = [transforms.Resize(size)]
    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


# def __scale_image(img, size, method=Image.BICUBIC):
#     w, h = size
#     return img.resize((w, h), method)


class Video(data.Dataset):
    def __init__(self, dataset):
        super(Video, self).__init__()
        # self.new_size = [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
        self.new_size = [219, 219]
        # self.num_frames = config.DATASET.NUM_FRAMES
        frame_steps = 32
        self.num_frames = 16 #consecutive
        # frame_steps = min(frame_steps, self.num_frames)
        # root = config.DATASET.ROOT
        # root = '../../datasets/'
        # dataset_name = config.DATASET.DATASET
        # dataset_name = 'avenue/'
        # train_set = config.DATASET.TRAINSET
        # train_set = 'training/'
        # lower_bound = 0
        # self.dir = os.path.join(root, dataset_name, train_set)
        self.dir = dataset
        assert (os.path.exists(self.dir))

        videos = self._colect_filelist(self.dir)

        converted_videos = []
        for video in videos:
            if isinstance(video, list):
                for file_path in video:
                    if file_path.endswith('.avi'):
                        converted_video, frame_count = self.convert_video_to_images(file_path)
                        if converted_video:
                            converted_videos.append(converted_video)
                    else:
                        # If not in video format, add as is
                        converted_videos.append(file_path)
            elif video.endswith('.avi'):
                converted_video, frame_count = self.convert_video_to_images(video)
                if converted_video:
                    converted_videos.append(converted_video)
            else:
                # If not in video format, add as is
                converted_videos.append(video)
        # for video in videos:
        #     # Check if the file is in video format (e.g., .avi)
        #     if video.endswith('.avi'):
        #         converted_video = self.convert_video_to_images(video)
        #         if converted_video:
        #             converted_videos.append(converted_video)
        #     else:
        #         # If not in video format, add as is
        #         converted_videos.append(video)

        self.videos = converted_videos
        

        split_videos = [[video[i:i + frame_count]
                         for i in range(0, len(video) // self.num_frames * self.num_frames,
                                        frame_steps)]
                        for video in videos]

        self.videos = []
        for video in split_videos:
            for sub_video in video:
                if len(sub_video) == self.num_frames:
                    self.videos.append(sub_video)

        self.num_videos = len(self.videos)

    def convert_video_to_images(self, video_path):
        # Use OpenCV to convert the video to images (e.g., jpg)
        output_folder = 'Images'  # You can specify the output folder
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        mask = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # image_path = os.path.join(output_folder, f"frame{frame_count:04d}.jpg")
            mog = cv2.createBackgroundSubtractorMOG2()
            foreground_mask = mog.apply(frame)
            # cv2.imwrite(image_path, frame)
            # images.append(image_path)
            mask.append(foreground_mask)

        cap.release()
        return mask, frame_count


    def _colect_filelist(self, root):
        include_ext = [".png", ".jpg", "jpeg", ".bmp", ".avi"]
        dirs = [x[0] for x in os.walk(root, followlinks=True)]

        dirs = natsort.natsorted(dirs)

        datasets = [
            [os.path.join(fdir, el) for el in natsort.natsorted(os.listdir(fdir))
             if os.path.isfile(os.path.join(fdir, el))
             and not el.startswith('.')
             and any([el.endswith(ext) for ext in include_ext])]
            for fdir in dirs
        ]

        return [el for el in datasets if el]

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        video_name = self.videos[index]
        raw_frames = [Image.open(f).convert('RGB') for f in video_name]

        video = []
        for f in raw_frames:
            transform = get_transform(self.new_size).float
            f = transform(f)
            mog = cv2.createBackgroundSubtractorMOG2()
            foreground_mask = mog.apply(f)
            video.append(foreground_mask)
            # video.append(f)
            

        return {'video': video, 'video_name': video_name}


class TestVideo(data.Dataset):
    def __init__(self, config):
        super(TestVideo, self).__init__()
        self.new_size = [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]]
        # root = config.DATASET.ROOT
        # dataset_name = config.DATASET.DATASET
        # test_set = config.DATASET.TESTSET
        # self.dir = os.path.join(root, dataset_name, test_set)
        self.num_frames = config.DATASET.NUM_FRAMES
        frame_steps = config.DATASET.FRAME_STEPS
        frame_steps = min(frame_steps, self.num_frames)
        lower_bound = config.DATASET.LOWER_BOUND
        self.dir = config.DATASET.TESTSET 
        assert (os.path.exists(self.dir))

        videos = self._colect_filelist(self.dir)

        converted_videos = []
        for video in videos:
            if isinstance(video, list):
                for file_path in video:
                    if file_path.endswith('.avi'):
                        converted_video = self.convert_video_to_images(file_path)
                        if converted_video:
                            converted_videos.append(converted_video)
                    else:
                        # If not in video format, add as is
                        converted_videos.append(file_path)
            elif video.endswith('.avi'):
                converted_video = self.convert_video_to_images(video)
                if converted_video:
                    converted_videos.append(converted_video)
            else:
                # If not in video format, add as is
                converted_videos.append(video)
        # for video in videos:
        #     # Check if the file is in video format (e.g., .avi)
        #     if video.endswith('.avi'):
        #         converted_video = self.convert_video_to_images(video)
        #         if converted_video:
        #             converted_videos.append(converted_video)
        #     else:
        #         # If not in video format, add as is
        #         converted_videos.append(video)
        split_videos = [[video[i:i + self.num_frames]
                         for i in range(0, len(video) // self.num_frames * self.num_frames,
                                        frame_steps if len(video) > lower_bound else 1)]
                        for video in videos]

        self.videos = []
        for video in split_videos:
            for sub_video in video:
                if len(sub_video) == self.num_frames:
                    self.videos.append(sub_video)


        self.videos = converted_videos

        self.num_videos = len(self.videos)

    def convert_video_to_images(self, video_path):
        # Use OpenCV to convert the video to images (e.g., jpg)
        output_folder = 'Images'  # You can specify the output folder
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        images = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            image_path = os.path.join(output_folder, f"frame{frame_count:04d}.jpg")
            cv2.imwrite(image_path, frame)
            images.append(image_path)
        
        cap.release()
        return images

    def _colect_filelist(self, root):
        include_ext = [".png", ".jpg", "jpeg", ".bmp", ".avi"]
        dirs = [x[0] for x in os.walk(root, followlinks=True)]

        dirs = natsort.natsorted(dirs)

        datasets = [
            [os.path.join(fdir, el) for el in natsort.natsorted(os.listdir(fdir))
             if os.path.isfile(os.path.join(fdir, el))
             and not el.startswith('.')
             and any([el.endswith(ext) for ext in include_ext])]
            for fdir in dirs
        ]

        return [el for el in datasets if el]

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        video_name = self.videos[index]
        # raw_frames = [Image.open(f).convert('RGB') for f in video_name]
        # raw_frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in video_name]
        # raw_frames = [cv2.imread(f) for f in video_name]
        # raw_frames = Image.open(name).convert('RGB')
        # raw_frames = []
        # for f in video_name:
        #     try:
        #         frame = [Image.open(f).convert('RGB') for f in video_name]
        #         # frame = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        #         if frame is not None:
        #             raw_frames.append(frame)
        #         else:
        #             print(f"Failed to load frame: {f}")
        #     except Exception as e:
        #         print(f"Error loading frame: {f}\n{str(e)}")
        video = []
        
        for f in video_name:
        # for f in video_name:
            frame = Image.open(f).convert('RGB')
            transform = get_transform(self.new_size)
            f = transform(frame)
            f = np.array(f)
            # f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
            # f = cv2.resize(f, (self.new_size[1], self.new_size[0]))  # Resize to (height, width)
            mog = cv2.createBackgroundSubtractorMOG2()
            # f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            foreground_mask = mog.apply(f)
            video.append(foreground_mask)
            # video.append(f)

        return {'video': video, 'video_name': video_name}
