import os.path
from util.sealingNails_npz import SealingNailDatasetNPZ
from util.data_util import collate_fn
from torch.utils.data import DataLoader
from model.sem.GraphAttention import graphAttention_seg_repro as Model
import logging
import torch
from tqdm import tqdm
import time


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('LeakyReLU') != -1:
        m.inplace = True


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def main():
    """Parameter"""
    # Change device selection to prefer CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = 'weight_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logs_dir = os.path.join('logs', log_path)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logs_dir = os.path.join(logs_dir, 'log_embedding.txt')
    file_handler = logging.FileHandler(logs_dir)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    root = './data/sealingNail_npz'
    class_label = {'Background1': 0, 'Burst': 1, 'Pit': 2, 'Stain': 3, 'Warpage': 4, 'Background2': 5, 'Burst2': 6, 'Pinhole': 7}
    feat_dim = 6
    num_class = 8
    num_points = 16384
    optimizer = 'Adam'
    learning_rate = 0.001
    weight_decay = 1e-04
    end_epoch = 300
    lr_decay = 0.5
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 20
    global_epoch = 0

    """Load Dataset"""
    TRAIN_SET = SealingNailDatasetNPZ(root=root, npoints=num_points, split='train', use_cache=True)
    # Reduce num_workers for better stability
    num_workers = min(4, os.cpu_count() or 1)
    trainDataLoader = DataLoader(TRAIN_SET, batch_size=4, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn, pin_memory=True)
    TEST_SET = SealingNailDatasetNPZ(root=root, npoints=num_points, split='test', use_cache=True)
    testDataLoader = DataLoader(TEST_SET, batch_size=4, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=collate_fn, pin_memory=True)

    """Calculate Weight"""
    weight = [float('{:.4f}'.format(i)) for i in TRAIN_SET.weight.tolist()]
    l_weight = [float('{:.4f}'.format(i)) for i in TRAIN_SET.l_weight.tolist()]
    logger.info('The weight of class: %s' % weight)
    logger.info('The weight of loss function: %s' % l_weight)
    weight = torch.tensor(TRAIN_SET.l_weight, dtype=torch.float).to(device)

    # Model Parameter
    classifier = Model(c=feat_dim, k=num_class)
    # Move model to device before applying any operations
    classifier = classifier.to(device)
    classifier.apply(inplace_relu)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    # loss_fn = FocalLoss(weight=weight, device=device)

    # load weight
    try:
        checkpoint = torch.load('./log/weight/last_model.pth', weights_only=True)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('use pretrain model')
    except:
        logger.info('no existing model, starting training from scratch...')
        start_epoch = 0

    # Optimizer Parameter
    if optimizer == 'Adam':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    best_mean_class_IoU = 0
    for epoch in range(start_epoch, end_epoch):
        logger.info('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, end_epoch))
        """Performance parameter"""
        train_intersection = torch.zeros(num_class, device=device)
        train_union = torch.zeros(num_class, device=device)
        train_targets = torch.zeros(num_class, device=device)
        train_loss = 0
        eval_intersection = torch.zeros(num_class, device=device)
        eval_union = torch.zeros(num_class, device=device)
        eval_targets = torch.zeros(num_class, device=device)
        eval_loss = 0

        '''Adjust learning rate and BN momentum'''
        lr = max(learning_rate * (lr_decay ** (epoch // MOMENTUM_DECCAY_STEP)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        logger.info('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (coords, feats, targets, offset) in tqdm(enumerate(trainDataLoader), desc=f'Epoch: {epoch + 1}/{end_epoch}', total=len(trainDataLoader), smoothing=0.9):
            # Remove these lines as they're redundant and cause the error
            # coords = coords.cuda()
            # feats = feats.cuda()
            # labels = labels.cuda()  # This line caused the error
            
            # Correct data movement to device
            coords, feats, targets, offset = coords.float().to(device), feats.float().to(device), targets.long().to(device), offset.to(device)
            
            """Training"""
            optimizer.zero_grad()
            seg_pred = classifier([coords, feats, offset])
            seg_pred = seg_pred.contiguous().view(-1, num_class)
            targets = targets.view(-1, 1)[:, 0]
            preds = seg_pred.data.max(1)[1]
            loss = loss_fn(seg_pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            """Training data statistics"""
            for label in class_label.values():
                train_intersection[label] += torch.logical_and(preds == label, targets == label).sum().data.cpu().item()
                train_union[label] += torch.logical_or(preds == label, targets == label).sum().data.cpu().item()
                train_targets[label] += torch.sum(targets == label).data.cpu().item()

        train_mAcc = (train_intersection / train_targets).mean().item()
        train_OA = train_intersection.sum() / train_targets.sum()
        train_IoU = train_intersection / train_union
        train_mIoU = train_IoU.mean().item()

        """print"""
        logger.info('Training loss is: %s %.5f' % (' ' * 12, (train_loss / len(trainDataLoader))))
        logger.info('Training overall accuracy: %s %.5f' % (' ' * 3, train_OA))
        logger.info('Training mean accuracy is: %s %.5f' % (' ' * 3, train_mAcc))
        for cat in class_label.keys():
            logger.info('Training mIoU of %s %f' % (cat + ' ' * (16 - len(cat)), train_IoU[class_label[cat]]))
        logger.info('Training mean class mIoU %s %f' % (' ' * 8, train_mIoU))

        if 1:
            with torch.no_grad():
                classifier = classifier.eval()
                """Validation"""
                for i, (coords, feats, targets, offset) in tqdm(enumerate(testDataLoader), desc='Validation', total=len(testDataLoader), smoothing=0.9):
                    coords, feats, targets, offset = coords.float().to(device), feats.float().to(device), targets.long().to(device), offset.to(device)
                    seg_pred = classifier([coords, feats, offset])
                    seg_pred = seg_pred.contiguous().view(-1, num_class)
                    targets = targets.view(-1, 1)[:, 0]
                    preds = seg_pred.data.max(1)[1]
                    loss = loss_fn(seg_pred, targets)
                    eval_loss += loss.item()
                    """Validation data statistics"""
                    for label in class_label.values():
                        eval_intersection[label] += torch.logical_and(preds == label, targets == label).sum().data.cpu().item()
                        eval_union[label] += torch.logical_or(preds == label, targets == label).sum().data.cpu().item()
                        eval_targets[label] += torch.sum(targets == label).data.cpu().item()

            eval_mAcc = (eval_intersection / eval_targets).mean().item()
            eval_OA = eval_intersection.sum() / eval_targets.sum()
            eval_IoU = eval_intersection / eval_union
            eval_mIoU = eval_IoU.mean().item()

            """print"""
            logger.info('Validation loss is: %s %.5f' % (' ' * 10, (eval_loss / len(testDataLoader))))
            logger.info('Validation overall accuracy: %s %.5f' % (' ', eval_OA))
            logger.info('Validation mean accuracy is: %s %.5f' % (' ', eval_mAcc))
            for cat in class_label.keys():
                logger.info('Validation mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), eval_IoU[class_label[cat]]))
            logger.info('Validation mean class mIoU %s %f' % (' ' * 6, eval_mIoU))

            logger.info('Save last model...')
            save_path = os.path.join('logs', log_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logger.info('Saving in %s' % save_path)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            last_model = os.path.join(save_path, 'last_model.pth')
            torch.save(state, last_model)
            logger.info('Completed.')
            if eval_mIoU >= best_mean_class_IoU:
                best_mean_class_IoU = eval_mIoU
                logger.info('Save best model...')
                save_path = os.path.join('logs', log_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                logger.info('Saving in %s' % save_path)
                state = {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                best_model = os.path.join(save_path, 'best_model.pth')
                torch.save(state, best_model)
                logger.info('Completed.')

        global_epoch += 1


if __name__ == "__main__":
    main()

