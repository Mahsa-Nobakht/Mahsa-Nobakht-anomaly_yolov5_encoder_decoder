import argparse,sys,os,torch,cv2,pickle,time
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import transforms as T
from torchvision import utils
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import multivariate_normal
from dataset import DAEDataset

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
sys.path.append('pytorch_yolov5/')
# from pytorch_yolov5.models.experimental import *
# from pytorch_yolov5.utils.datasets import *
# from pytorch_yolov5.utils.utils import *
from pytorch_yolov5.utils.torch_utils import select_device, time_synchronized


def detect(save_img=False):    
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    sample_path="output/test_samples"
    os.makedirs(sample_path,exist_ok=True)
    # device = torch_utils.select_device(opt.device)
    device = select_device(opt.device)
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    # load model
    ae_model = torch.load(opt.DAE_ckpt,map_location="cuda")
    ae_model = ae_model.to('cuda')
    ae_model.eval()


    dataset = DAEDataset(source)


    # frames_folder = source
    # # Path to the folder containing frames
    # frames_folder = 'path_to_frames_folder'

    # # Get a list of all frame files in the folder
    # frame_files = [f for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))]

    # Get a list of all frame files in the folder
    # frame_files = [f for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))]
    anomaly_flag_list = []  # Create a list to store anomaly flags for each frame

    # Loop through each frame file
    # for frame_file in frame_files:
    #     # Read the frame
    #     frame_path = os.path.join(frames_folder, frame_file)
    #     frame = cv2.imread(frame_path)

        # Resize frame to 219x219
        # resized_frame = cv2.resize(frame, (219, 219))
    
    frame_index_list = []  # Added to store frame indices
    reg_score_list = []  # Added to store regularity scores
        
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    loss_dae = []
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # for path, img, im0s, vid_cap in dataset:
    for i, img in enumerate(dataset):
    # for img in frame_files:
        # img = torch.from_numpy(img).to(device)
        img = img.to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img.float()
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = torch_utils.time_synchronized()
        reg_score = torch.zeros(img.size(0)) # shape (N)
        anomaly_flag = torch.zeros(img.size(0)) # shape (N)
        frame_index = torch.arange(img.size(0)) # shape (N)
        # t1 = time_sync()
        # pred = model(img, augment=opt.augment)[0]
        pred, loss = ae_model(img)
        # Sum over the channels, height, and width dimensions
        # loss = loss.sum()
        loss_dae.append(loss)

        # Normalize the error to get the regularity score
        # reg_score[i] = 1 - loss
        reg_score = loss 

        frame_index_list.append(i)
        reg_score_list.append(reg_score.cpu().item())  # Move to CPU before storing

        # Set the anomaly flag to 1 if the regularity score is below a threshold
        # if reg_score[i] < 0.5:
        #     anomaly_flag[i] = 1

        # if reg_score < 0.5:
        #     anomaly_flag = 1
        anomaly_flag_list.append(1 if reg_score < 0.5 else 0)

    reg_score_list = np.array(reg_score_list)  # Convert to NumPy array

    anomaly_threshold = 0.4
    anomaly_indices = np.where(reg_score_list > anomaly_threshold)[0]
    # Plot the regularity score as a line plot
    plt.plot(frame_index_list, reg_score_list, color='blue')
    plt.ylim(0, 1)
    # Plot the red bound around the line
    # plt.fill_between(frame_index_list, reg_score_list - 0.05, reg_score_list + 0.05, color='red', alpha=0.5)
    # Plot the pink shaded area where the anomaly flag is 1
    plt.fill_between(frame_index_list, 0, 1, where=np.array(anomaly_flag_list) == 1, color='pink', alpha=0.5)
    # plt.scatter(frame_index_list[anomaly_indices], reg_score_list[anomaly_indices], color='red', marker='o', label='Anomaly > 40%')
    # plt.scatter(frame_index_list[anomaly_indices.astype(int)], reg_score_list[anomaly_indices.astype(int)], color='red', marker='o', label='Anomaly > 40%')

    # Set the x-axis and y-axis labels
    plt.xlabel('Frame index')
    plt.ylabel('Regularity score')
    # # Add the images and the red box to the plot
    # You can adjust the position and size of the images and the box as you like
    # Here we use the first and the last frames as examples
    # plt.annotate('', xy=(0, 0.5), xytext=(200, 0.5), arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    # plt.imshow(img[0].cpu().permute(1, 2, 0), extent=(200, 400, 0, 0.5))
    # plt.annotate('', xy=(1000, 0.5), xytext=(800, 0.5), arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))
    # plt.imshow(img[-1].cpu().permute(1, 2, 0), extent=(800, 1000, 0, 0.5))
    # plt.plot([900, 950], [0.1, 0.1], color='red', linewidth=3)
    # plt.plot([900, 950], [0.4, 0.4], color='red', linewidth=3)
    # plt.plot([900, 900], [0.1, 0.4], color='red', linewidth=3)
    # plt.plot([950, 950], [0.1, 0.4], color='red', linewidth=3)
    # Save the plot as an image file
    plt.savefig('reg_score_plot.png')
    # while True:
    #     cv2.imshow("Prediction", pred[0])
    #     cv2.waitKey(0)
    #     sys.exit()
    # cv2.destroyAllWindows()
    # img = img[0].cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    # plt.imshow(img[10])
    # plt.title('MoG')
            
    plt.show()   


    print('Done. (%.3fs)' % (time.time() - t0))

    # return loss_dae, img

# def process_video_or_images(video_or_images_path, output_folder, ae_model):
#     frames = []
    
#     if video_or_images_path.endswith('.mp4') or video_or_images_path.endswith('.avi'):
#         cap = cv2.VideoCapture(video_or_images_path)
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         for _ in range(frame_count):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(frame)
#         cap.release()
#     else:
#         frame_files = [f for f in os.listdir(video_or_images_path) if os.path.isfile(os.path.join(video_or_images_path, f))]
#         for frame_file in frame_files:
#             frame_path = os.path.join(video_or_images_path, frame_file)
#             frame = cv2.imread(frame_path)
#             frames.append(frame)

    
#     # Autoencoder
#     anomaly_scores, _ = detect(frames, ae_model)
    
#     # Display or save in Output folder
#     if output_folder:
#         os.makedirs(output_folder, exist_ok=True)
#         for i, frame in enumerate(frames):
#             
#             frame_with_mog = cv2.bitwise_and(frame, frame, mask=binary_masks[i])
            
#             cv2.imwrite(os.path.join(output_folder, f'frame_{i}_score_{anomaly_scores[i]:.4f}.png'), frame_with_mog)
#     else:
#         # Display results
#         for i, frame in enumerate(frames):
#             cv2.imshow('Frame with MOG Mask', cv2.bitwise_and(frame, frame, mask=binary_masks[i]))
#             print(f'Frame {i} - Anomaly Score: {anomaly_scores[i]}')
#             cv2.waitKey(0)
#         cv2.destroyAllWindows()


# def calculate_AUC(labels, predictions):
#     ground_truth = np.zeros(predictions.shape[0])
#     for anomalous_frames in labels:
#         ground_truth[anomalous_frames] = 1
    
#     auc_score = roc_auc_score(ground_truth, predictions)
#     return auc_score

# def plot_anomaly_scores(anomaly_scores):
#     plt.plot(range(len(anomaly_scores)), anomaly_scores, color='blue', lw=2)
#     plt.xlabel('Frame Number')
#     plt.ylabel('Anomaly Score')
#     plt.title('Anomaly Score over Frames')
#     plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='pytorch_yolov5/weights/yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--DAE_ckpt', type=str, default='/home/mahsa/gmm_dae/output/checkpoint/ped1/Train_img_50.pt')
    # parser.add_argument('--DIDAE_ckpt', type=str, default='output/checkpoint/train_dimg_100.pt')
    parser.add_argument('--source', type=str, default='/home/mahsa/gmm_dae/dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test003', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output/pkl', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--min_boxsize', type=int, default=10, help='box area less than it will be ignored')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    os.makedirs(opt.output, exist_ok=True)
    with torch.no_grad():
        # MSE_FEA_List,SI_fea,DI_fea = detect()
        # Loss, img = detect()
        detect()
    # Loss_np = np.array(Loss)
    # Loss_np = np.array([item.cpu().numpy() for item in Loss])

    # print("img shape:", img.shape, "img type:", type(img))
    # print("Loss shape:", Loss_np.shape, "Loss type:", type(Loss_np))

    # video_or_images_path = 'path/to/your/video_or_images_folder_or_file'
    # output_folder = 'path/to/your/output_folder'
    # process_video_or_images(video_or_images_path, output_folder, ae_model)

    
    # labels = np.load('path/to/your/ground_truth.npy')  
    # auc_score = calculate_AUC(labels, anomaly_scores)
    # print(f'AUC Score: {auc_score}')

    # plot_anomaly_scores(anomaly_scores)

    # mean_loss = np.mean(Loss)
    # std_loss = np.std(Loss)
    # threshold = mean_loss + 2 * std_loss
    # if len(Loss) > 0:
    #     # mean_loss = np.mean(Loss)
    #     mean_loss = np.mean([item.cpu().numpy() for item in Loss])

    #     # std_loss = np.std(Loss)
    #     std_loss = np.std([item.cpu().numpy() for item in Loss])
    #     threshold = mean_loss + 2 * std_loss
    #     anomalous_frames = [1 if error > threshold else 0 for error in Loss]
    # else:
    #     print("No losses recorded during detection.")

    # # arr=np.load('dataset/frame_labels_ped2.npy')
    # # ground_th=arr[0]
    # if sum(anomalous_frames) == 0:
    #     print("No anomalies in the ground truth.")
    # else:
    #     # Calculate AUC
    #     fpr, tpr, thresholds = roc_curve(anomalous_frames, Loss_np)
    #     roc_auc = auc(fpr, tpr)
    #     print("DEA AUC: ", roc_auc)




    # # Calculate Anomaly Score based on reconstruction errors
    # anomaly_scores = []  # Store anomaly scores for each frame
    # for error in reconstruction_errors:
    #     anomaly_score = 1 - (error / max_error)  # Normalize error to [0, 1] range
    #     anomaly_scores.append(anomaly_score)

    # # Create a plot of Anomaly Score
    # plt.plot(range(len(anomaly_scores)), anomaly_scores, color='blue', lw=2)
    # plt.xlabel('Frame Number')
    # plt.ylabel('Anomaly Score')
    # plt.title('Anomaly Score over Frames')
    # plt.show()


    # anomalous_frames = [1 if error > threshold else 0 for error in Loss_np]
    # fpr, tpr, thresholds = roc_curve(anomalous_frames, Loss_np)
    # fpr, tpr, thresholds = roc_curve(img.cpu().numpy(), [l.cpu().numpy() for l in Loss])
    # fpr, tpr, thresholds = roc_curve(anomalous_frames.cpu().numpy(), [l.cpu().numpy() for l in Loss])

    # roc_auc = auc(fpr, tpr)


# # Define a threshold for pixel-level anomaly detection (at least 40% of pixels being anomalous)
# threshold_percentage = 0.4  # Set the desired percentage (40% in this case)
# anomaly_mask = (error_map > threshold).astype(np.uint8)

# # Check if at least 40% of pixels are anomalous
# if np.mean(anomaly_mask) > threshold_percentage:
#     anomaly_pixel_map = cv2.bitwise_or(anomaly_pixel_map, anomaly_mask)


