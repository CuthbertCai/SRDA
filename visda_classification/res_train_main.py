from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
import torch.nn.functional as F
import os
from attack import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=1, metavar='K',
                    help='hyper parameter only for virtual adversarial attack')
parser.add_argument('--iterations', type=int, default=1, metavar='K',
                    help='hyper parameter for optimizing attack loss')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='save/vatDA', metavar='B',
                    help='board dir')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save model parameters')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--train_path', type=str, default='data/VisDA/train/', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='data/VisDA/validation/', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--resnet', type=str, default='101', metavar='B',
                    help='which resnet 18,50,101,152,200')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='N',
                    help='hyper parameter for attack')
parser.add_argument('--attack', type=str, default='fgsm', metavar='N',
                    help='choose which attack algorithm to attack')
parser.add_argument('--no_adapt', type=str, default='No', metavar='N',
                    help='train model without adaptation')
parser.add_argument('--zeta', type=float, default=1e-6, metavar='N',
                    help='hyper parameter for VATAttack')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
save_path = args.save+'/'+args.attack
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_epoch = args.save_epoch

data_transforms = {
    train_path: transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path,val_path]}
dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
print ('classes'+str(dset_classes))
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path],dsets[val_path],batch_size)
dataset = train_loader.load_data()
test_loader = CVDataLoader()
opt= args
test_loader.initialize(dsets[train_path],dsets[val_path],batch_size,shuffle=False)
dataset_test = test_loader.load_data()
option = 'resnet'+args.resnet

num_classes = 12
class_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle',
              'person', 'plant', 'skateboard', 'train', 'truck']
# G = ResBase(option)
# F1 = ResClassifier(num_layer=num_layer)
# F2 = ResClassifier(num_layer=num_layer)
# F1.apply(weights_init)
# F2.apply(weights_init)
if args.no_adapt is not 'No':
    P = ResNet(option)
    net_init(P, 'kaiming')
else:
    G = ResBase(option)
    C = ResClassifier(num_layer=num_layer)

lr = args.lr
if args.cuda:
    # G.cuda()
    # F1.cuda()
    # F2.cuda()
    if args.no_adapt is not 'No':
        P.cuda()
    else:
        G.cuda()
        C.cuda()

if args.no_adapt is not 'No':
    if args.optimizer == 'momentum':
        optimizer_P = optim.SGD(P.parameters(),momentum=0.9,lr=args.lr,weight_decay=0.0005)
    elif args.optimizer == 'adam':
        optimizer_P = optim.Adam(P.parameters(), lr=args.lr,weight_decay=0.0005)
    else:
        optimizer_P = optim.Adadelta(P.parameters(),lr=args.lr,weight_decay=0.0005)
else:
    if args.optimizer == 'momentum':
        optimizer_G = optim.SGD(G.parameters(),momentum=0.9,lr=args.lr,weight_decay=0.0005)
        optimizer_C = optim.SGD(C.parameters(),momentum=0.9,lr=args.lr,weight_decay=0.0005)
    elif args.optimizer == 'adam':
        optimizer_G = optim.Adam(G.parameters(), lr=args.lr,weight_decay=0.0005)
        optimizer_C = optim.Adam(C.parameters(), lr=args.lr,weight_decay=0.0005)
    else:
        optimizer_G = optim.Adadelta(G.parameters(),lr=args.lr,weight_decay=0.0005)
        optimizer_C = optim.Adadelta(C.parameters(),lr=args.lr,weight_decay=0.0005)

def update_attacker(attack):
    if attack == 'fgsm':
        attacker = FGSMAttack(model=C, epsilon=args.epsilon)
    elif attack == 'vat':
        attacker = VATAttack(model=C, epsilon=args.epsilon, zeta=args.zeta, num_k=args.num_k)
    elif attack == 'ifgsm':
        attacker = IFGSMAttack(model=C, epsilon=args.epsilon, momentum=args.momentum, num_k=args.num_k)
    else:
        raise Exception('Choose an adversarial attack algorithm.')

    return attacker

def ent(output):
    return torch.mean(torch.sum(-output * torch.log(output + 1e-6),1))

def abs_discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

def square_discrepancy(out1, out2):
    return torch.mean((out1 - out2)**2)

def js_discrepancy(out1, out2):
    out = 0.5 * (F.softmax(out1, dim=1) + F.softmax(out2, dim=1))
    return ent(out) - 0.5 * (ent(F.softmax(out1, dim=1)) + ent(F.softmax(out2, dim=1)))

def ce_discrepancy(out1, out2):
    return - torch.mean(F.softmax(out2, dim=1) * torch.log(F.softmax(out1, dim=1) + 1e-6))

def ce_discrepancy_weights(out1, out2, weights):
    return - torch.sum(F.softmax(out2, dim=1) * torch.log(F.softmax(out1, dim=1) + 1e-6) * weights)

def train(num_epoch):
    print(save_path)
    loss_rcd = []
    accuracy_rcd = []
    best_acc = 0.0
    best_G = None
    best_C = None
    criterion = nn.CrossEntropyLoss().cuda()
    for ep in range(num_epoch):
        G.train()
        C.train()
        for batch_idx, data in enumerate(dataset):
            if batch_idx * batch_size > 30000:
                break
            if args.cuda:
                data1 = data['S']
                target1 = data['S_label']
                data2  = data['T']
                target2 = data['T_label']
                data1, target1 = data1.cuda(), target1.cuda()
                data2, target2 = data2.cuda(), target2.cuda()
            data = Variable(torch.cat((data1,data2),0))
            target1 = Variable(target1)
            img_s = data[:batch_size,...]
            img_t = data[batch_size:,...]

            optimizer_G.zero_grad()
            optimizer_C.zero_grad()
            # supervised crossentropy loss of source domain
            output = C(G(data))

            output_s = output[:batch_size,:]
            output_t = output[batch_size:,:]
            output_t = F.softmax(output_t, 1)

            entropy_loss = - torch.mean(torch.log(torch.mean(output_t, 0) + 1e-6))

            loss_s = criterion(output_s, target1) + 0.02 * entropy_loss
            loss_s.backward()
            optimizer_G.step()
            optimizer_C.step()
            optimizer_G.zero_grad()
            optimizer_C.zero_grad()

            for _ in range(args.iterations):
                # unsupervised adversarial attack loss of source and target domain
                # generate adversarial samples
                feature = G(data)
                feature_s, feature_t = feature[:batch_size,...], feature[batch_size:,...]
                output = C(feature)
                output_s, output_t = output[:batch_size,...], output[batch_size:,...]
                attacker = update_attacker(args.attack)
                if args.attack == 'fgsm':
                    adversarial_s = attacker.perturb(feature_s.data.cpu().numpy(), target1.data.cpu().numpy())
                    adversarial_t = attacker.perturb(feature_t.data.cpu().numpy(),
                                                     F.softmax(output_t, dim=1).data.cpu().numpy())
                elif args.attack == 'vat':
                    adversarial_s = attacker.perturb(feature_s.data.cpu().numpy())
                    adversarial_t = attacker.perturb(feature_t.data.cpu().numpy())
                elif args.attack == 'ifgsm':
                    adversarial_s = attacker.perturb(feature_s.data.cpu().numpy(), target1.data.cpu().numpy())
                    adversarial_t = attacker.perturb(feature_t.data.cpu().numpy(),
                                                     F.softmax(output_t, dim=1).data.cpu().numpy())
                else:
                    raise Exception('Choose an adversarial attack algorithm.')

                adversarial_s = Variable(torch.from_numpy(adversarial_s).cuda())
                adversarial_t = Variable(torch.from_numpy(adversarial_t).cuda())

                # entropy = torch.sum(F.softmax(output_t, dim=1)*torch.log(F.softmax(output_t, dim=1)+1e-6), dim=1)
                # weights = F.softmax(entropy, dim=0).unsqueeze(1)

                output_ads = C(adversarial_s)
                output_adt = C(adversarial_t)

                # loss_attack = ce_discrepancy(output_adt, output_t) + ce_discrepancy(output_ads, output_s)
                loss_attack = ce_discrepancy(output_adt, output_t)
                # loss_attack = ce_discrepancy_weights(output_adt, output_t, weights)
                loss_attack.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()
                loss_rcd.append(loss_attack.data[0])

            if batch_idx % args.log_interval == 0:
                print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss_s: {:.6f}\tLoss_attack: {:.6f} Entropy: {:.6f}'.format(
                    ep, batch_idx * len(data), 70000,
                    100. * batch_idx / 70000, loss_s.data[0],loss_attack.data[0],entropy_loss.data[0]))
            if batch_idx == 1 and ep >1:
                accuracy = test(ep)
                accuracy_rcd.append(accuracy)
                if accuracy > best_acc:
                    best_acc =  accuracy
                    best_G = G
                    best_C = C
                G.train()
                C.train()
    if args.save_model:
        torch.save(best_G.state_dict(),
                '%s/iter%s_num_k%s_epoch%s_resnet%s_accuracy%s_G.pt' % (save_path, str(args.iterations), str(num_k), num_epoch, args.resnet, str(best_acc)))
        torch.save(best_C.state_dict(),
                '%s/iter%s_num_k%s_epoch%s_resnet%s_accuracy%s_C.pt' % (save_path, str(args.iterations), str(num_k), num_epoch, args.resnet, str(best_acc)))
        np.save('%s/loss_epoch%s_accuracy%s.npy' % (save_path, num_epoch, str(best_acc)), loss_rcd)
        np.save('%s/accuracy_epoch%s_accuracy%s.npy' % (save_path, num_epoch, str(best_acc)), accuracy_rcd)


def train_noadapt(epoch):
    criterion = nn.CrossEntropyLoss().cuda()
    P.train()
    torch.cuda.manual_seed(1)

    for batch_idx, data in enumerate(dataset):
        img_s = data['S']
        label_s = data['S_label']
        if img_s.size()[0] < batch_size:
            break
        img_s = img_s.cuda()
        label_s = Variable(label_s.long().cuda())

        img_s = Variable(img_s)
        P.zero_grad()

        # supervised crossentropy loss of source domain
        output = P(img_s)

        loss = criterion(output, label_s)

        loss.backward()
        optimizer_P.step()

        if batch_idx > 500:
            return batch_idx

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), 70000,
                    float(100. * batch_idx) / 70000, loss.data[0]))
    return batch_idx

def test(epoch, record_file=None, save_model=False):
    G.eval()
    C.eval()
    test_loss = 0
    stest_loss = 0
    correct = 0
    scorrect = 0
    size = 0
    ssize = 0
    per_class_num = np.zeros((num_classes))
    per_class_correct = np.zeros((num_classes)).astype(np.float32)
    for batch_idx, data in enumerate(dataset_test):
        img = data['T']
        label = data['T_label']
        simg = data['S']
        slabel = data['S_label']
        img, simg, label, slabel = img.cuda(), simg.cuda(), label.long().cuda(), slabel.long().cuda()
        img, simg, label, slabel = Variable(img, volatile=True), Variable(simg, volatile=True), \
                                   Variable(label), Variable(slabel)
        output = C(G(img))
        soutput = C(G(simg))
        test_loss += F.cross_entropy(output, label).data[0]
        stest_loss += F.cross_entropy(soutput, slabel).data[0]
        pred = output.data.max(1)[1]
        spred = soutput.data.max(1)[1]
        k = label.data.size()[0]
        sk = slabel.data.size()[0]
        correct += pred.eq(label.data).cpu().sum()
        scorrect += spred.eq(slabel.data).cpu().sum()
        size += k
        ssize += sk
        pred = pred.cpu().numpy()
        for t in range(num_classes):
            t_ind = np.where(label.data.cpu().numpy() == t)
            correct_ind = np.where(pred[t_ind[0]]==t)
            per_class_correct[t] += float(len(correct_ind[0]))
            per_class_num[t] += float(len(t_ind[0]))
    test_loss = test_loss / size
    stest_loss = stest_loss / ssize
    per_class_accuracy = per_class_correct / per_class_num
    print(
        '\nTarget Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)) \n'.format(
            test_loss, correct, size,
            float(100. * correct) / size))
    print(
        '\nSource Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)) \n'.format(
            stest_loss, scorrect, ssize,
            float(100. * scorrect) / ssize))
    for ind, category in enumerate(class_list):
        print('%s:%s' %(category, per_class_accuracy[ind]))
    if save_model and epoch % save_epoch == 0:
        torch.save(G.state_dict(),
                    '%s/iter%s_num_k%s_epoch%s_resnet%s_accuracy%s_G.pt' % (save_path, str(args.iterations), str(num_k), epoch, args.resnet, str(100. * correct / size)))
        torch.save(C.state_dict(),
                   '%s/iter%s_num_k%s_epoch%s_resnet%s_accuracy%s_C.pt' % (save_path, str(args.iterations), str(num_k), epoch, args.resnet, str(100. * correct / size)))
    if record_file:
        record = open(record_file, 'a')
        print('recording %s', record_file)
        record.write('%s\n' % (float(correct) / size))
        record.close()
    accuracy = float(100. * correct / size)
    return accuracy

def test_noadapt(epoch):
    P.eval()
    test_loss_t = 0
    test_loss_s = 0
    correct_s = 0
    correct_t = 0
    size_t = 0
    size_s = 0
    for batch_idx, data in enumerate(dataset_test):
        img = data['T']
        label = data['T_label']
        simg = data['S']
        slabel = data['S_label']
        img, simg, label, slabel = img.cuda(), simg.cuda(), label.long().cuda(), slabel.long().cuda()
        img, simg, label, slabel = Variable(img, volatile=True), Variable(simg, volatile=True), \
                                   Variable(label), Variable(slabel)
        output_t = P(img)
        output_s = P(simg)
        test_loss_t += F.cross_entropy(output_t, label).data[0]
        test_loss_s += F.cross_entropy(output_s, slabel).data[0]
        pred_t = output_t.data.max(1)[1]
        pred_s = output_s.data.max(1)[1]
        k = label.data.size()[0]
        sk = slabel.data.size()[0]
        correct_t += pred_t.eq(label.data).cpu().sum()
        correct_s += pred_s.eq(slabel.data).cpu().sum()
        size_t += k
        size_s += sk
    test_loss_t = test_loss_t / size_t
    test_loss_s = test_loss_s / size_s
    print(
        '\nTarget Test set: Average loss: {:.4f}, Accuracy : {}/{} ({:.3f}%)) \n'.format(
            test_loss_t, correct_t, size_t,
            float(100. * correct_t) / size_t))
    print(
        '\nSource Test set: Average loss: {:.4f}, Accuracy : {}/{} ({:.3f}%)) \n'.format(
            test_loss_s, correct_s, size_s,
            float(100. * correct_s) / size_s))


if __name__ == '__main__':
    if args.no_adapt is not 'No':
        for epoch in range(args.epochs+1):
            train_noadapt(epoch)
            if epoch % 10 == 0:
                test_noadapt(epoch)
    else:
        train(args.epochs+1)
