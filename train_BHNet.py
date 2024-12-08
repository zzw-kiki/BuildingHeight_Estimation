import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import yaml
import shutil
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils import data
from ptsemseg.models import get_model
from ptsemseg.utils import get_logger
from tensorboardX import SummaryWriter
from ptsemseg.loader.diy_dataset_BHNet import dataloaderbh
from ptsemseg.loader.diyloader_BHNet import myImageFloder
from ptsemseg.loader.diyloader_BHNet import myImageFloder_HR
from ptsemseg.loader.diy_dataset_BHNet import dataloaderbh_HR
import torch.nn.functional as F

def main(cfg, writer, logger):
    start_time = time.time()  # 记录开始时间
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device(cfg["training"]["device"])

    # Setup Dataloader
    data_path = cfg["data"]["path"]
    n_classes = cfg["data"]["n_class"]
    n_maxdisp = cfg["data"]["n_maxdisp"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]

    # Load dataset
    trainimg, trainlab, valimg, vallab = dataloaderbh(data_path)
    traindataloader = torch.utils.data.DataLoader(
        myImageFloder(trainimg, trainlab, True),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    testdataloader = torch.utils.data.DataLoader(
        myImageFloder(valimg, vallab),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Setup Model
    model = get_model(cfg["model"], n_maxdisp=n_maxdisp, n_classes=n_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # print the model
    start_epoch = 0
    resume = cfg["training"]["resume"]
    if os.path.isfile(resume):
        if os.path.isfile(resume):
            logger.info(f"Loading checkpoint '{resume}'")
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})")
            start_epoch = checkpoint['epoch']
        else:
            logger.info("No checkpoint found at resume. Will start from scratch.")

    # define task-dependent log_variance
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)
    # log_var_c = torch.tensor(1.) # fix the weight of semantic segmentation
    log_var_c = torch.zeros((1,), requires_grad=True)
    log_var_d = torch.zeros((1,), requires_grad=True)

    # get all parameters (model parameters + task dependent log variances)
    params = ([p for p in model.parameters()] + [log_var_a] + [log_var_b] + [log_var_c]
              + [log_var_d])
    logger.info(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    optimizer = torch.optim.Adam(params, lr=cfg["training"]["learning_rate"], betas=(0.9, 0.999))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam(params, lr=cfg["training"]["learning_rate"], betas=(0.9, 0.999))

    criterion = 'rmse' #useless

    for epoch in range(epochs-start_epoch):
        epoch = start_epoch + epoch
        adjust_learning_rate(optimizer, epoch)
        model.train()
        train_loss = list()
        train_mse_final = 0.
        train_mse_S1 = 0.
        train_mse_S2 = 0.
        train_mse_POI = 0.
        count = 0
        print_count = 0
        vara = list()
        varb = list()
        varc = list()
        vard = list()
        # with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for x, y_true in tqdm(traindataloader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            ypred1, ypred2, ypred3 , ypred4 = model.forward(x)
            mse_S1 = F.mse_loss(ypred1, y_true, reduction='mean').cpu().detach().numpy()
            mse_S2 = F.mse_loss(ypred2, y_true, reduction='mean').cpu().detach().numpy()
            mse_POI = F.mse_loss(ypred3, y_true, reduction='mean').cpu().detach().numpy()
            loss_mse = F.mse_loss(ypred4 , y_true, reduction='mean').cpu().detach().numpy()
            loss = loss_weight([ypred1, ypred2, ypred3,ypred4],
                               [y_true],
                               [log_var_a.to(device), log_var_b.to(device), log_var_c.to(device),
                                log_var_d.to(device)])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()
            train_loss.append(loss.cpu().detach().numpy())
            train_mse_final += loss_mse*x.shape[0]
            train_mse_S1 += mse_S1*x.shape[0]
            train_mse_S2 += mse_S2*x.shape[0]
            train_mse_POI += mse_POI*x.shape[0]

            count += x.shape[0]

            vara.append(log_var_a.cpu().detach().numpy())
            varb.append(log_var_b.cpu().detach().numpy())
            varc.append(log_var_c.cpu().detach().numpy())
            vard.append(log_var_d.cpu().detach().numpy())


            if print_count%20 ==0:
                print('training loss %.3f, rmse %.3f, S1 %.2f, S2 %.2f, POI %.2f, final %.2f'%
                  (loss.item(), np.sqrt(loss_mse), log_var_a, log_var_b, log_var_c, log_var_d))
            print_count += 1

        train_final_rmse = np.sqrt(train_mse_final/count)
        train_S1_rmse = np.sqrt(train_mse_S1 / count)
        train_S2_rmse = np.sqrt(train_mse_S2 / count)
        train_POI_rmse = np.sqrt(train_mse_POI / count)
        # test
        val_rmse = test_epoch(model, criterion,
                              testdataloader, device, epoch)
        # logger.info(f'Epoch {epoch} RMSE: train {train_final_rmse:.3f}, test {val_rmse:.3f}')
        print("epoch %d rmse: train_final %.3f, test_final %.3f, train_S1 %.3f, train_S2 %.3f, train_POI %.3f" %
              (epoch, train_final_rmse, val_rmse,train_S1_rmse,train_S2_rmse,train_POI_rmse))

        # save models
        if epoch % 2 == 0: # every five internval
            savefilename = os.path.join(logdir, 'finetune_'+str(epoch)+'.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': np.mean(train_loss),
                'test_loss': np.mean(val_rmse), #*100
            }, savefilename)
        #
        writer.add_scalar('train loss',
                          (np.mean(train_loss)), #average
                          epoch)
        writer.add_scalar('train rmse',
                          (np.mean(train_final_rmse)), #average
                          epoch)
        writer.add_scalar('val rmse',
                          (np.mean(val_rmse)), #average
                          epoch)
        writer.add_scalar('weight S1',
                          (np.mean(vara)), #average
                          epoch)
        writer.add_scalar('weight S2',
                          (np.mean(varb)),  # average
                          epoch)
        writer.add_scalar('weight POI',
                          (np.mean(varc)),  # average
                          epoch)
        writer.add_scalar('weight final_height',
                          (np.mean(vard)),  # average
                          epoch)
        writer.close()
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算总耗时
        logger.info(f"训练完成，总耗时: {elapsed_time:.2f} 秒")
        # print(f"训练总耗时: {elapsed_time:.2f} 秒")
        torch.cuda.empty_cache()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = cfg["training"]["learning_rate"]
    elif epoch <=250:
        lr = cfg["training"]["learning_rate"] * 0.1
    elif epoch <=300:
        lr = cfg["training"]["learning_rate"] * 0.01
    else:
        lr = cfg["training"]["learning_rate"] * 0.025 # 0.0025 before
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr #added

def loss_weight(y_pred, y_true, log_vars):
  #loss 0 S1 height
  precision0 = torch.exp(-log_vars[0])
  diff0 = F.mse_loss(y_pred[0],y_true[0],reduction='mean')
  loss0 = diff0*precision0 + log_vars[0]
  #loss 1 S2 height
  precision1 = torch.exp(-log_vars[1])
  diff1 = F.mse_loss(y_pred[1], y_true[0], reduction='mean')
  loss1 = diff1*precision1 + log_vars[1]
  #loss 2 POI height
  precision2 = torch.exp(-log_vars[2])
  diff2 = F.mse_loss(y_pred[2], y_true[0], reduction='mean')
  loss2 = diff2*precision2 + log_vars[2]
  # loss 3 final height
  precision3 = torch.exp(-log_vars[3])
  diff3 = F.mse_loss(y_pred[3], y_true[0], reduction='mean')
  loss3 = diff3 * precision3 + log_vars[3]
  return loss0+loss1+loss2+loss3

def test_epoch(model, criterion, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        losses = 0.
        count = 0
        for x, y_true in tqdm(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True)

            y_pred = model.forward(x)
            lossv = F.mse_loss(y_pred, y_true, reduction='mean').cpu().detach().numpy()
            losses += lossv*x.shape[0]
            count += x.shape[0]

        lossfinal = np.sqrt(losses/count)
        # print('test error %.3f rmse' % lossfinal)
        torch.cuda.empty_cache()

        return lossfinal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/BHNet.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], "BHNet_10m_14%_300epoch")
    writer = SummaryWriter(log_dir=logdir)
    # logger = setup_logger(logdir)
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Starting training with config: %s", args.config)

    main(cfg, writer, logger)
