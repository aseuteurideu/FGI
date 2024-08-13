import argparse, re, pickle
from model import My_Network
from dataset_3d import *
from resnet_2d3d import neq_load_customized
from utils import denorm, AverageMeter, save_checkpoint, write_log

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils import data
import torch.utils.data
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--net', default='resnet18', choices=['resnet18'], type=str)
parser.add_argument('--final_dim', default=1024, type=int, help='length of vector output from audio/video subnetwork')
parser.add_argument('--spatial_size', default=28, type=int, help='spatial size')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume training')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--out_dir', default='output', type=str, help='Output directory containing Deepfake_data')
parser.add_argument('--print_freq', default=10, type=int, help='frequency of printing output during training')
parser.add_argument('--hyper_param', default=0.99, type=float, help='margin hyper parameter used in loss equation')
parser.add_argument('--test', default='', type=str)
parser.add_argument('--dropout', default=0.5, type=float)

parser.add_argument('--dataset', type=str, default='dfdc', choices=['dfdc', 'fakeavceleb'], help='Dataset selection')

parser.add_argument('--hyperparam_search', action='store_true', help='Hyperparameter (lr, batch size, weight decay) search')

parser.add_argument('--save_all', action='store_true', help='save all epoch')

#################################### with pseudo fake
parser.add_argument('--aud_min_fake_len', default=2, type=float, help='Minimum fake length for audio. If between 0 to 1 means the ratio with data length.')
parser.add_argument('--aud_max_fake_len', default=-1, type=float, help='Maximum fake length for audio. -1: data length. If between 0 to 1 means the ratio with data length.')

parser.add_argument('--vis_min_fake_len', default=2, type=float, help='Minimum fake length for visual. If between 0 to 1 means the ratio with data length.')
parser.add_argument('--vis_max_fake_len', default=-1, type=float, help='Maximum fake length for visual. -1: data length. If between 0 to 1 means the ratio with data length.')
##############################
parser.add_argument('--using_pseudo_fake', action='store_true', help='Using pseudo fake even though still using the real-fake')
##############################
parser.add_argument('--with_att', action='store_true', help='use attention')
parser.add_argument('--residual_conn', action='store_true', help='use residual connection in the attention')
##############################################################################

def load_rawaudio_data(transform, args, mode='train'):

    dataset = deepfake_3d_rawaudio(out_dir=args.out_dir, mode=mode,
                                   transform=transform,
                                   vis_min_fake_len=args.vis_min_fake_len, vis_max_fake_len=args.vis_max_fake_len,
                                   aud_min_fake_len=args.aud_min_fake_len, aud_max_fake_len=args.aud_max_fake_len,
                                   using_pseudo_fake=args.using_pseudo_fake, dataset_name=args.dataset)
    return dataset

def get_rawaudio_data(transform, args, mode='train'):
    print('Loading data for "%s" ...' % mode)
    dataset = load_rawaudio_data(transform, args, mode)

    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=my_collate_rawaudio)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      collate_fn=my_collate_rawaudio)

    else:
        print('No mode ', mode)
        sys.exit()
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_tmp/v6_mydf-'
        if args.dataset != 'dfdc':
            exp_path += args.dataset + '-'
        if args.spatial_size != 28:
            exp_path += 'spas' + str(args.spatial_size) + '-'
        if args.with_att:
            if args.residual_conn:
                exp_path += 'rc-'
        if args.using_pseudo_fake:
            exp_path += 'upf-'


        exp_path += '{args.img_dim}_{0}_bs{args.batch_size}_lr{1}'.format(
            'r%s' % args.net[6::], \
            args.old_lr if args.old_lr is not None else args.lr, \
            args=args)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    print('exp_path:', exp_path)
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path, os.path.split(exp_path)[-1]



def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    global cuda
    cuda = torch.device('cuda')

    model=My_Network(with_attention=args.with_att, residual_conn=args.residual_conn,
                     spatial_size=args.spatial_size)

    model = model.cuda()
    model = nn.DataParallel(model)
    global criterion
    criterion = nn.BCELoss()

    print('\n===========Check Grad============ ')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    least_loss = 0
    global iteration
    iteration = 0

    if args.test:
        if os.path.isfile(args.test):
            if args.with_att:
                if args.residual_conn:
                    assert 'rc-' in args.test
                else:
                    assert 'rc-' not in args.test
                if args.spatial_size != 28:
                    assert 'spas' + str(args.spatial_size) + '-' in args.test
                else:
                    assert 'spas' not in args.test

            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Use non-equal load ==')
                sys.exit()
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
            global num_epoch;
            num_epoch = checkpoint['epoch']
        elif args.test == 'random':
            print("=> [Warning] loaded random weights")
        else:
            raise ValueError()

        # test result save folder
        paths = args.test.split('/')
        test_split = 'balance' if args.dataset == 'fakeavceleb' else 'imbalance'
        save_folder = os.path.join('test_results', paths[1], paths[-1][:-8], test_split)
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        transform = transforms.Compose([
            Scale(size=(args.img_dim, args.img_dim)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_loader = get_rawaudio_data(transform, args, 'test')

        global test_pred
        test_pred = {}
        global test_target
        test_target = {}
        global test_number_ofile_number_of_chunks
        test_number_ofile_number_of_chunks = {}

        test(test_loader, model, args, save_folder=save_folder)

        dataset = '' if args.dataset == 'dfdc' else args.dataset
        file_pred = open(os.path.join(save_folder, dataset + "file_pred.pkl"), "wb")
        pickle.dump(test_pred, file_pred)
        file_pred.close()
        file_target = open(os.path.join(save_folder, dataset + "file_target.pkl"), "wb")
        pickle.dump(test_target, file_target)
        file_target.close()
        file_number_of_chunks = open(os.path.join(save_folder, dataset + "file_number_of_chunks.pkl"), "wb")
        pickle.dump(test_number_ofile_number_of_chunks, file_number_of_chunks)
        file_number_of_chunks.close()

        del test_pred
        del test_target
        del test_number_ofile_number_of_chunks

        sys.exit()
    else:  # not test
        torch.backends.cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)/', args.resume).group(1))
            print(args.old_lr)
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            least_loss = checkpoint['least_loss']
            try:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('=> [Warning]: weight structure is not equal to checkpoint; Use non-equal load ==')
                model = neq_load_customized(model, checkpoint['state_dict'])
                sys.exit(1)
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))
            sys.exit(1)

    transform = transforms.Compose([
        Scale(size=(args.img_dim, args.img_dim)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # make it -1 to 1 instead of using imagenet statistics
    ])

    train_loader = get_rawaudio_data(transform, args, 'train')

    global total_sum_scores_real;
    total_sum_scores_real = 0
    global total_sum_scores_fake;
    total_sum_scores_fake = 0
    global count_real;
    count_real = 0
    global count_fake;
    count_fake = 0
    # setup tools
    global de_normalize;
    de_normalize = denorm(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    global img_path;
    img_path, model_path, _ = set_path(args)
    global writer_train
    try:  # old version
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except:  # v1.7
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, real_distance, fake_distance = train(train_loader, model, optimizer, epoch, args)

        writer_train.add_scalar('global/loss', train_loss, epoch)
        writer_train.add_scalar('global/real_distance', real_distance, epoch)
        writer_train.add_scalar('global/fake_distance', fake_distance, epoch)

        # save check_point
        if epoch == 0:
            least_loss = train_loss
        is_best = train_loss <= least_loss
        least_loss = min(least_loss, train_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'net': args.net,
            'state_dict': model.state_dict(),
            'least_loss': least_loss,
            'optimizer': optimizer.state_dict(),
            'iteration': iteration
        }, is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch + 1)), keep_all=args.save_all)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train(data_loader, model, optimizer, epoch, args):
    losses = AverageMeter()
    real_distances = AverageMeter()
    fake_distances = AverageMeter()
    att_losses = AverageMeter()
    dist_losses = AverageMeter()
    model.train()
    global iteration
    for idx, (video_seq, audio_seq, target, audiopath) in enumerate(data_loader):
        target = 1-target  # flip label. 0: real. 1: fake

        tic = time.time()
        video_seq = video_seq.to(cuda)
        audio_seq = audio_seq.to(cuda)
        target = target.to(cuda)
        B = video_seq.size(0)

        out, vid_out_dist, att = model(video_seq, audio_seq)

        del video_seq
        del audio_seq

        loss = criterion(out, target.float())  # BCE loss

        losses.update(loss.item(), B)

        number_of_real = torch.sum(target.view(-1) == 0).detach().cpu().item()
        if number_of_real > 0:
            real_distances.update(torch.mean(vid_out_dist[target.view(-1) == 0]).item())
        number_of_fake = torch.sum(target.view(-1) == 1).detach().cpu().item()
        if number_of_fake > 0:
            fake_distances.update(torch.mean(vid_out_dist[target.view(-1) == 1]).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time iter: {3}\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'.format(
                epoch, idx, len(data_loader), time.time() - tic,
                loss=losses))

            total_weight = 0.0
            decay_weight = 0.0
            for m in model.parameters():
                if m.requires_grad: decay_weight += m.norm(2).data.cuda(0)
                total_weight += m.norm(2).data.cuda(0)
            print('Decay weight / Total weight: %.3f/%.3f' % (decay_weight, total_weight))

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/real_distance', real_distances.val, iteration)
            writer_train.add_scalar('local/fake_distance', fake_distances.val, iteration)
            writer_train.add_scalar('local/att_loss', att_losses.val, iteration)
            writer_train.add_scalar('local/dist_loss', dist_losses.val, iteration)

            iteration += 1

    return losses.avg, real_distances.avg, fake_distances.avg


def test(data_loader, model, args, save_folder='test_results'):
    losses = AverageMeter()
    real_distances = AverageMeter()
    fake_distances = AverageMeter()
    model.eval()

    with torch.no_grad():
        pred_fake = []
        pred_real = []
        dist_fake = []
        dist_real = []

        for idx, (video_seq, audio_seq, target, audiopath) in tqdm(enumerate(data_loader), total=len(data_loader)):
            if len(video_seq) == 0:
                continue

            target = 1 - target  # flip label. 0: real. 1: fake

            # just forward the first data in the batch
            video_seq = video_seq[0].unsqueeze(0).to(cuda)
            audio_seq = audio_seq[0].unsqueeze(0).to(cuda)
            target = target[0].unsqueeze(0).to(cuda)
            B = video_seq.size(0)
            assert B == 1

            pred, vid_aud_dist, att = model(video_seq, audio_seq)

            del video_seq
            del audio_seq

            tar = target[0, :].view(-1).item()

            vid_name = audiopath[0].split('/')[-2]
            # print(vid_name)
            if (test_pred.get(vid_name)):
                test_pred[vid_name] += pred[0].view(-1).item()
                test_number_ofile_number_of_chunks[vid_name] += 1
            else:
                test_pred[vid_name] = pred[0].view(-1).item()
                test_number_ofile_number_of_chunks[vid_name] = 1

            if (test_target.get(vid_name)):
                pass
            else:
                test_target[vid_name] = tar

            if tar == 1:
                pred_fake.append(pred[0].item())
                dist_fake.append(torch.mean(vid_aud_dist[0]).item())
                fake_distances.update(torch.mean(vid_aud_dist[0]).item())
            else:
                pred_real.append(pred[0].item())
                dist_real.append(torch.mean(vid_aud_dist[0]).item())
                real_distances.update(torch.mean(vid_aud_dist[0]).item())

    print('Loss {loss.avg:.4f}\t'.format(loss=losses))
    print('Fake_distances {:0.4f}\t'.format(fake_distances.avg))
    print('Real_distances {:0.4f}\t'.format(real_distances.avg))
    write_log(content='Loss {loss.avg:.4f}\t'.format(loss=losses, args=args),
              epoch=num_epoch,
              filename=os.path.join(os.path.dirname(args.test), 'test_log.md'))
    return losses.avg






if __name__ == '__main__':
    args = parser.parse_args()
    args.old_lr = None

    main(args)
