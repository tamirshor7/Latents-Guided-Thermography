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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#We initialize some colors for segmentation results display
RED = np.array([255, 0, 0])
GREEN = np.array([0, 255, 0])
BLUE = np.array([0, 0, 255])
YELLOW = RED + GREEN
CYAN = GREEN + BLUE
PINK = RED + BLUE
BLACK = np.zeros(3)
SKIN = RED + GREEN * 0.8 + BLUE * 0.65
ID_TO_COLORS = {1: RED, 2: GREEN, 3: BLUE, 4: YELLOW, 5: CYAN, 6: PINK, 7: BLACK, 8: SKIN}

IM_SIZE_HEAT = (120,160) #we need these constants to initialize the linear layer
IM_SIZE_GRAY = (480,640)

class ClassificationModel(torch.nn.Module):
    def __init__(self, in_features=128,gray = False):
        super(ClassificationModel, self).__init__()

        self.unet =  UNet(in_features, 1, bilinear=True)
        self.project = torch.nn.Linear(IM_SIZE_GRAY[0]*IM_SIZE_GRAY[1] if gray else IM_SIZE_HEAT[0]*IM_SIZE_HEAT[1],1)


    def forward(self, x):
        return self.project(self.unet(x).reshape(x.shape[0],-1))

    def save_weights(self, model_save_path: str) -> None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        return

    def load_weights(self, model_save_path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(model_save_path, map_location=device))
        return


def run(config,gray_encoder, gray_decoder, encoder_ckpt_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set, val_set, _ = \
        prepare_dataset(config=config, mode='train',binary=True)

    num_image_channel= 1 if gray_encoder else 3 #data image channel number

    # Build the decoder model
    decoder = ClassificationModel(gray=gray_decoder).to(device)

    # Build the encoder model
    encoder = CUTSEncoder(
        in_channels=num_image_channel,
        num_kernels=config.num_kernels,
        random_seed=config.random_seed,
        sampled_patches_per_image=config.sampled_patches_per_image).to(device)

   
    encoder.load_weights(f"{encoder_ckpt_root}/dmr_runs{'_gray' if gray_encoder else ''}/checkpoints/dmr_run.pty", device=device)
    log('CUTSEncoder: Model weights successfully loaded.', to_console=True)

    optimizer = torch.optim.Adam(decoder.parameters(),
                                 lr=0.005)

    loss_fn_classification = torch.nn.BCEWithLogitsLoss().to(device)

    best_val_loss = float('inf')



    for epoch_idx in tqdm(range(config.max_epochs)):

        classification_loss = 0
        abs_error = 0
        abs_amount = 0
        decoder.train()
        iters_num = 0
        for iter, (x_train, y_train) in enumerate(train_set):
            B = x_train.shape[0]

            x_train = x_train.type(torch.FloatTensor).to(device)
            if gray_encoder and not gray_decoder: #we diverge the extra channels to the batch dimension since encoder expects single channel, but data is 3 channels
                x_train = x_train.reshape(B * 3, 1, IM_SIZE_HEAT[0], IM_SIZE_HEAT[1])

            elif gray_encoder != gray_decoder:
                x_train = x_train.repeat(1, 3, 1, 1) #we need data to have 3 channels for the encoder, so we just repeat the single channel


            y_train = y_train.type(torch.float32).to(device)


            with torch.no_grad(): #we do not train the encoder with the decoder
                z, _, _, _, _ = encoder(x_train)

            class_pred = decoder(z)
            if gray_encoder and not gray_decoder: #cancel repeat by averaging (we could've settled this with the decoder's
                                                  #number of output channels - averaging worked well enough and provides
                                                  #grounds for comparison
                class_pred = class_pred.reshape(B, 3, 1).mean(dim=1)

            loss = loss_fn_classification(class_pred, y_train.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                #calculate absolute error (accuracy) in classification
                train_pred = (torch.nn.Sigmoid()(class_pred.view(-1,)))>0.5
                abs_error += (((torch.nn.Sigmoid()(class_pred.view(-1,)))>0.5)!=y_train).sum()
                abs_amount += class_pred.shape[0]

            classification_loss += loss.item()
            iters_num += 1

        train_loss_classifier = classification_loss / iters_num #avg classification loss


        log('Train [%s/%s] classifier loss: %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss_classifier),
            filepath=config.log_dir,
            to_console=False)

        log('Train [%s/%s] Abs loss: %.3f out of %.3f'
            % (epoch_idx + 1, config.max_epochs, abs_error, abs_amount),
            filepath=config.log_dir,
            to_console=False)

        #evaluation loop

        decoder.eval()
        encoder.eval()

        val_loss = 0
        abs_error_val = 0
        abs_amount_val = 0
        bs = config.batch_size
        all_preds = None
        all_gts = None
        with torch.no_grad():
            iters_num = 0
            for i in range(0,len(train_set.dataset.dataset.dataset.test_images),bs):

                if i<len(train_set.dataset.dataset.dataset.test_images)-bs:
                    x_val = train_set.dataset.dataset.dataset.test_images[i:i+bs]
                    y_val = train_set.dataset.dataset.dataset.test_labels[i:i + bs]
                else: #handle batch remainder
                    x_val = train_set.dataset.dataset.dataset.test_images[i:]
                    y_val = train_set.dataset.dataset.dataset.test_labels[i:]

                B = x_val.shape[0]

                x_val = torch.Tensor(x_val).cuda()
                y_val = torch.Tensor(y_val).cuda()
                x_val = x_val.type(torch.FloatTensor).to(device)
                y_val = y_val.type(torch.float32).to(device)

                if gray_encoder and not gray_decoder:
                    x_val = x_val.reshape(B * 3, 1, IM_SIZE_HEAT[0], IM_SIZE_HEAT[1])

                elif gray_encoder != gray_decoder:
                    x_val = x_val.repeat(1, 3, 1, 1)

                z, _, _, _, _ = encoder(x_val)
                class_pred = decoder(z)
                if gray_encoder and not gray_decoder:
                    class_pred = class_pred.reshape(B, 3, 1).mean(dim=1)

                val_loss += loss_fn_classification(class_pred.view(-1,), y_val)
                iters_num += 1

                with torch.no_grad():
                    all_preds = torch.nn.Sigmoid()(class_pred.view(-1, )).view(-1,1) > 0.5 if all_preds is None else  torch.cat((all_preds,(torch.nn.Sigmoid()(class_pred.view(-1, ))).view(-1,1) > 0.5))
                    all_gts = y_val if all_gts is None else torch.cat((all_gts,y_val))
                    abs_error_val += (((torch.nn.Sigmoid()(class_pred.view(-1, ))) > 0.5) != y_val).sum()
                    abs_amount_val += class_pred.shape[0]



        val_loss = val_loss / iters_num
        y_pred = all_preds.cpu().detach().numpy()
        y_true = all_gts.cpu().detach().numpy()
        precision = precision_score(y_true, y_pred)

        # Compute recall
        
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        # Compute F1 score
        f1 = f1_score(y_true, y_pred)
        log(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}. Accuracy: {accuracy}",to_console=False)
        log('Validation [%s/%s] classification loss: %.3f'
            % (epoch_idx + 1, config.max_epochs, val_loss),
            filepath=config.log_dir,
            to_console=False)
        log('Validation [%s/%s] Abs loss: %.3f out of %.3f'
            % (epoch_idx + 1, config.max_epochs, abs_error_val,abs_amount_val),
            filepath=config.log_dir,
            to_console=False)

        abs_error_val = abs_error_val.item()
        if abs_error_val< best_val_loss:
            best_val_loss = abs_error_val
            decoder.save_weights(config.model_save_path)
            log('CUTSEncoder: Model weights successfully saved.',
                filepath=config.log_dir,
                to_console=False)

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
    cfg_file_name = f"dmr_classification_{'g' if args.gray_encoder else 'h'}_enc_{'g' if args.gray_decoder else 'h'}_dec.yaml"
    config_path = f"{args.cfg_path}/{cfg_file_name}"
    config = AttributeHashmap(yaml.safe_load(open(config_path)))
    config.config_file_name = cfg_file_name
    config = parse_settings(config, log_settings=False)
    seed_everything(config.random_seed)

    run(config=config, gray_encoder=args.gray_encoder,gray_decoder=args.gray_decoder, encoder_ckpt_root=args.encoder_ckpt_root)
