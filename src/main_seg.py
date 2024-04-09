import argparse
import numpy as np
import torch
import yaml
from data_utils.prepare_dataset import prepare_dataset
from model import CUTSEncoder, UNet
from tqdm import tqdm
from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log
from utils.parse import parse_settings
from utils.seed import seed_everything
import os
from matplotlib import pyplot as plt
from torchmetrics.functional.classification import multiclass_jaccard_index as mji
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

#We initialize some colors for segmentation results display
NUM_CLASSES = 8 #two nipples, two breasts, two armpits, neck and non-ROI
RED = np.array([255, 0, 0])
GREEN = np.array([0, 255, 0])
BLUE = np.array([0, 0, 255])
YELLOW = RED + GREEN
CYAN = GREEN + BLUE
PINK = RED + BLUE
BLACK = np.zeros(3)
SKIN = RED + GREEN * 0.8 + BLUE * 0.65

ID_TO_COLORS = {1: RED, 2: GREEN, 3: BLUE, 4: YELLOW, 5: CYAN, 6: PINK, 7: BLACK, 8: SKIN}

IM_SIZE = (120,160)


class SegmentationModel(torch.nn.Module):
    def __init__(self, in_features=128, num_classes=NUM_CLASSES):
        super(SegmentationModel, self).__init__()
        self.num_classes = num_classes
        self.unet = UNet(in_features, num_classes, bilinear=True)

    def forward(self, x):
        return self.unet(x).permute(0, 2, 3, 1).reshape(-1, self.num_classes)


    def save_weights(self, model_save_path: str) -> None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        return

    def load_weights(self, model_save_path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(model_save_path, map_location=device))
        return


def plot_seg(config, img, desc_label, gray=False, best=False, root=None):
    '''Plot Segmentation masks of an [H,W,Num_Classes] image'''
    if root is None:
        root = "seg_plots"
    os.makedirs(f"{root}", exist_ok=True)
    os.makedirs(f"{root}/logs", exist_ok=True)
    os.makedirs(f"{root}//logs/vis_results_iter_{desc_label.split('_')[-1]}",
                exist_ok=True)
    if best:
        os.makedirs(
            f"{root}//logs/best_vis_results_iter_{desc_label.split('_')[-1]}",
            exist_ok=True)
    segmented_image_batch = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
    for i in range(img.shape[-1]):
        segmented_image_batch[torch.where(img.argmax(dim=-1).cpu().detach() == i)] = ID_TO_COLORS[i + 1]

    for idx, segmented_image in enumerate(segmented_image_batch):
        plt.imsave(
            f"{root}//logs/{'best_' if best else ''}vis_results_iter_{desc_label.split('_')[-1]}/{desc_label}_{idx}.png",
            segmented_image.astype(np.uint8) if gray else (
                np.kron(segmented_image, np.ones((4, 4, 1))).astype(np.uint8)))


def run(config,gray_encoder, gray_decoder, encoder_ckpt_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set, val_set, _ = \
        prepare_dataset(config=config, mode='train')
    num_image_channel = 1 if gray_encoder else 3  # data image channel number
    decoder = SegmentationModel().to(device)
    early_stop = 30  # num of unchanged epochs for early stop

    # Build the CUTA encoder model
    encoder = CUTSEncoder(
        in_channels=num_image_channel,
        num_kernels=config.num_kernels,
        random_seed=config.random_seed,
        sampled_patches_per_image=config.sampled_patches_per_image).to(device)
    encoder.load_weights(f"{encoder_ckpt_root}/dmr_runs{'_gray' if gray_encoder else ''}/checkpoints/dmr_run.pty",
                         device=device)

    log('CUTSEncoder: Model weights successfully loaded.', to_console=True)

    optimizer = torch.optim.Adam(decoder.parameters(),
                                 lr=0.005)

    loss_fn_segmentation = torch.nn.CrossEntropyLoss(torch.Tensor([1, 1, 2, 2, 1, 1, 1, 1]).to(device)) #we give higher weight to nipples to compensate for class imbalance

    best_val_loss = np.inf
    plot_coeff = 4 if gray_decoder else 1 #we use this to scale the sizes in segmentation plots
    non_improve_streak = 0 #for early stopping
    for epoch_idx in tqdm(range(config.max_epochs)):

        classification_loss = 0
        decoder.train()
        iters_num = 0
        for iter, (x_train, y_train, _) in enumerate(train_set):
            B = x_train.shape[0]

            # for non expanded
            x_train = x_train.type(torch.FloatTensor).to(device)
            if gray_encoder and not gray_decoder:
                x_train = x_train.reshape(B * 3, 1, IM_SIZE[0],IM_SIZE[1]) #encoder expects 1 channel but we have 3, so divert to batch dimension
            y_train = y_train if gray_decoder else torch.nn.AvgPool2d(4)(y_train) #downsample the labels if we use heatmap data, since our labels are only on grayscale
            y_train = y_train.type(torch.float32).to(device).permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
            with torch.no_grad():
                z, _, _, _, _ = encoder(x_train)



            class_pred = decoder(z)
            if gray_encoder and not gray_decoder:
                # we average over the channel dimension
                class_pred = class_pred.reshape(B,3,IM_SIZE[0],IM_SIZE[1],NUM_CLASSES).mean(dim=1).reshape(-1,NUM_CLASSES)

            loss = loss_fn_segmentation(class_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            classification_loss += loss.item()
            iters_num += 1

            #plot segmentation results
            plot_seg(config, class_pred.reshape(-1, IM_SIZE[0] * plot_coeff, IM_SIZE[1] * plot_coeff, NUM_CLASSES), f"train pred_iter_{iter}",
                     gray=gray_decoder)
            plot_seg(config, y_train.reshape(-1, IM_SIZE[0] * plot_coeff, IM_SIZE[1] * plot_coeff, NUM_CLASSES), f"train label_iter_{iter}",
                     gray=gray_decoder)

        train_loss_classifier = classification_loss / iters_num


        log('Train [%s/%s] classifier loss: %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss_classifier),
            filepath=config.log_dir,
            to_console=False)

        classification_loss = 0
        val_loss = 0
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            iters_num = 0
            for iter, (x_val, y_val, _) in enumerate(val_set):
                B = x_val.shape[0]


                y_val = y_val if gray_decoder else torch.nn.AvgPool2d(4)(y_val)
                x_val = x_val.type(torch.FloatTensor).to(device)
                if gray_encoder and not gray_decoder:
                    x_val = x_val.reshape(B * 3, 1, IM_SIZE[0],IM_SIZE[1])  # encoder expects 1 channel but we have 3, so divert to batch dimension
                y_val = y_val.type(torch.float32).to(device).permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)

                z, _, _, _, _ = encoder(
                    x_val)

                class_pred = decoder(z)

                if gray_encoder and not gray_decoder:
                    # we average over the channel dimension
                    class_pred = class_pred.reshape(B, 3, IM_SIZE[0], IM_SIZE[1], NUM_CLASSES).mean(dim=1).reshape(-1,
                                                                                                         NUM_CLASSES)

                val_loss += loss_fn_segmentation(class_pred, y_val)
                iters_num += 1
                class_pred = torch.nn.Softmax(dim=-1)(class_pred.reshape(-1, IM_SIZE[0] * plot_coeff, IM_SIZE[1] * plot_coeff, NUM_CLASSES))
                plot_seg(config, class_pred, f"test pred_iter_{iter}", gray=gray_decoder)
                plot_seg(config, y_val.reshape(-1, IM_SIZE[0] * plot_coeff, IM_SIZE[1] * plot_coeff, NUM_CLASSES), f"test label_{iter}",
                         gray=gray_decoder)

        val_loss = val_loss / iters_num

        jscore = mji(class_pred.reshape(-1, NUM_CLASSES).argmax(1), y_val.reshape(-1, NUM_CLASSES, IM_SIZE[0], IM_SIZE[1]).reshape(-1, NUM_CLASSES).argmax(dim=1),
                     NUM_CLASSES)
        preds_t, yt_t = class_pred.reshape(-1, NUM_CLASSES).argmax(1), y_val.reshape(-1, NUM_CLASSES, IM_SIZE[0]\
                                                        , IM_SIZE[1]).reshape(-1, NUM_CLASSES).argmax(dim=1)
        accuracy = (preds_t == yt_t).sum() / preds_t.shape[0]

        precision = MulticlassPrecision(num_classes=NUM_CLASSES).cuda()(preds_t, yt_t)
        recall = MulticlassRecall(num_classes=NUM_CLASSES).cuda()(preds_t, yt_t)
        f1 = MulticlassF1Score(num_classes=NUM_CLASSES).cuda()(preds_t, yt_t)

        log(f"mIoU: {jscore}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}", to_console=False)

        log('Validation [%s/%s] classification loss: %.3f'
            % (epoch_idx + 1, config.max_epochs, val_loss),
            filepath=config.log_dir,
            to_console=False)

        if val_loss < best_val_loss:
            non_improve_streak = 0
            best_val_loss = val_loss
            decoder.save_weights(config.model_save_path)
            log('CUTSEncoder: Model weights successfully saved.',
                filepath=config.log_dir,
                to_console=False)
            plot_seg(config, class_pred, f"test pred_iter_{iter}", gray=gray_decoder, best=True)
            plot_seg(config, y_val.reshape(-1, IM_SIZE[0] * plot_coeff, IM_SIZE[1] * plot_coeff, NUM_CLASSES), f"test label_{iter}", gray=gray_decoder,
                     best=True)
        else:
            non_improve_streak += 1

        if non_improve_streak >= early_stop:
            print("Early Stopping")
            break

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gray_encoder', action='store_true',
                        help="If set, use encoder trained on grayscale data. Else use encoder trained on heatmap data.")
    parser.add_argument('--gray_decoder', action='store_true',
                        help="If set, train decoder on grayscale data. Else use heatmap data for training.")
    parser.add_argument('--cfg-path', type=str, default="config/",
                        help="Path to config folder")

    parser.add_argument('--encoder-ckpt-root', type=str, default="../",
                        help="Path to root containing encoder checkpoint folders")
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    cfg_file_name = f"dmr_seg_{'g' if args.gray_encoder else 'h'}_enc_{'g' if args.gray_decoder else 'h'}_dec.yaml"
    config_path = f"{args.cfg_path}/{cfg_file_name}"
    config = AttributeHashmap(yaml.safe_load(open(config_path)))
    config.config_file_name = cfg_file_name
    config = parse_settings(config, log_settings=False)
    seed_everything(config.random_seed)

    run(config=config, gray_encoder=args.gray_encoder,gray_decoder=args.gray_decoder, encoder_ckpt_root=args.encoder_ckpt_root)
