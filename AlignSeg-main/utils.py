import time
from typing import Tuple, List, Dict

import faiss
import torch
import numpy as np
import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from skimage.measure import label
from collections import deque, defaultdict

from torch import nn
from torchvision import transforms
from torchvision.transforms import GaussianBlur
from torchmetrics import Metric


def save_checkpoint(state, is_best=0, gap=1, filename='models/checkpoint.pth', keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(os.path.dirname(filename),
                                   'epoch%s.pth' % str(state['epoch'] - gap))
    if not keep_all:
        try:
            os.remove(last_epoch_path)
        except:
            pass
    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth'))
        for i in past_best:
            try:
                os.remove(i)
            except:
                pass
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth' % str(state['epoch'])))


class PredsmIoU(Metric):
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """

    def __init__(self,
                 num_pred_classes: int,
                 num_gt_classes: int):
        """
        :param num_pred_classes: The number of predicted classes.
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_pred_classes = num_pred_classes
        self.num_gt_classes = num_gt_classes
        self.add_state("iou", [])
        self.add_state("iou_excludeFirst", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor, many_to_one=True, precision_based=True, linear_probe=False):
        pred = pred.cpu().numpy().astype(int)
        gt = gt.cpu().numpy().astype(int)
        assert len(np.unique(pred)) <= self.num_pred_classes
        assert np.max(pred) <= self.num_pred_classes
        iou_all, iou_excludeFirst = self.compute_miou(gt, pred, self.num_pred_classes, len(np.unique(gt)),
                                            many_to_one=many_to_one, precision_based=precision_based, linear_probe=linear_probe)
        self.iou.append(iou_all)
        self.iou_excludeFirst.append(iou_excludeFirst)

    def compute(self):
        """
        Compute mIoU
        """
        mIoU = np.mean(self.iou)
        mIoU_excludeFirst = np.mean(self.iou_excludeFirst)
        print('---mIoU computed---', mIoU)
        print('---mIoU exclude first---', mIoU_excludeFirst)
        return mIoU

    def compute_miou(self, gt: np.ndarray, pred: np.ndarray, num_pred: int, num_gt: int,
                     many_to_one=False, precision_based=False, linear_probe=False):
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt
        """
        assert pred.shape == gt.shape
        print(f"unique semantic class = {np.unique(gt)}")
        gt_class = np.unique(gt).tolist()
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        iou = [0] * num_gt

        if linear_probe:
            reordered_preds = pred
        else:
            if many_to_one:
                match = self._original_match(num_pred, num_gt, pred, gt, precision_based=precision_based)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, matched_preds in match.items():
                    for pred_i in matched_preds:
                        reordered_preds[pred == int(pred_i)] = int(target_i)
            else:
                match = self._hungarian_match(num_pred, num_gt, pred, gt)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, pred_i in zip(*match):
                    reordered_preds[pred == int(pred_i)] = int(target_i)
                # merge all unmatched predictions to background
                for unmatched_pred in np.delete(np.arange(num_pred), np.array(match[1])):
                    reordered_preds[pred == int(unmatched_pred)] = 0

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = (gt == gt_class[i_part])
            tmp_pred = (reordered_preds == gt_class[i_part])
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            iou[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        print('\tiou = ', iou, np.mean(iou[1:]))
        if len(iou) > 1:
            return np.mean(iou), np.mean(iou[1:])
        else:
            # return np.mean(iou), tp, fp, fn, reordered_preds.astype(int).tolist()
            return np.mean(iou), np.mean(iou)

    @staticmethod
    def get_score(flat_preds: np.ndarray, flat_targets: np.ndarray, c1: int, c2: int, precision_based: bool = False) \
            -> float:
        """
        Calculates IoU given gt class c1 and prediction class c2.
        :param flat_preds: flattened predictions
        :param flat_targets: flattened gt
        :param c1: ground truth class to match
        :param c2: predicted class to match
        :param precision_based: flag to calculate precision instead of IoU.
        :return: The score if gt-c1 was matched to predicted c2.
        """
        tmp_all_gt = (flat_targets == c1)
        tmp_pred = (flat_preds == c2)
        tp = np.sum(tmp_all_gt & tmp_pred)
        fp = np.sum(~tmp_all_gt & tmp_pred)
        if not precision_based:
            fn = np.sum(tmp_all_gt & ~tmp_pred)
            jac = float(tp) / max(float(tp + fp + fn), 1e-8)
            return jac
        else:
            prec = float(tp) / max(float(tp + fp), 1e-8)
            # print('\tgt, pred = ', c1, c2, ' | precision=', prec)
            return prec

    def compute_score_matrix(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray,
                             precision_based: bool = False) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        # print("Parallelizing iou computation")
        # start = time.time()
        score_mat = Parallel(n_jobs=self.n_jobs)(delayed(self.get_score)(pred, gt, c1, c2, precision_based=precision_based)
                                                 for c2 in range(num_pred) for c1 in np.unique(gt))
        # print(f"took {time.time() - start} seconds")
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T

    def _hungarian_match(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray):
        # do hungarian matching. If num_pred > num_gt match will be partial only.
        iou_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt)
        match = linear_sum_assignment(1 - iou_mat)
        print("Matched clusters to gt classes:")
        print(match)
        return match

    def _original_match(self, num_pred, num_gt, pred, gt, precision_based=False) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt, precision_based=precision_based)
        gt_class = np.unique(gt).tolist()
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_i in range(num_gt):
                score = score_mat[gt_i, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_class[gt_i]
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k, v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        # print('original match:', gt_to_matches)
        return gt_to_matches


class PredsmIoUKmeans(PredsmIoU):
    """
    Used to track k-means cluster correspondence to ground-truth categories during fine-tuning.
    """

    def __init__(self,
                 clustering_granularities: List[int],
                 num_gt_classes: int,
                 pca_dim: int = 50):
        """
        :param clustering_granularities: list of clustering granularities for embeddings
        :param num_gt_classes: number of ground-truth classes
        :param pca_dim: target dimensionality of PCA
        """
        super(PredsmIoU, self).__init__(compute_on_step=False, dist_sync_on_step=False)  # Init Metric super class
        self.pca_dim = pca_dim
        self.num_pred_classes = clustering_granularities
        self.num_gt_classes = num_gt_classes
        self.add_state("masks", [])
        self.add_state("embeddings", [])
        self.add_state("gt", [])
        self.n_jobs = -1  # num_jobs = num_cores
        self.num_train_pca = 4000000  # take num_train_pca many vectors at max for training pca

    def update(self, masks: torch.Tensor, embeddings: torch.Tensor, gt: torch.Tensor) -> None:
        self.masks.append(masks)
        self.embeddings.append(embeddings)
        self.gt.append(gt)

    def compute(self, is_global_zero: bool) -> List[any]:
        if is_global_zero:
            # interpolate embeddings to match ground-truth masks spatially
            embeddings = torch.cat([e.cpu() for e in self.embeddings], dim=0)  # move everything to cpu before catting
            valid_masks = torch.cat(self.masks, dim=0).cpu().numpy()
            res_w = valid_masks.shape[2]
            embeddings = nn.functional.interpolate(embeddings, size=(res_w, res_w), mode='bilinear')
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(valid_masks.shape[0] * res_w ** 2, -1).numpy()

            # Normalize embeddings and reduce dims of embeddings by PCA
            normalized_embeddings = (embeddings - np.mean(embeddings, axis=0)) / (
                    np.std(embeddings, axis=0, ddof=0) + 1e-5)
            d_orig = embeddings.shape[1]
            pca = faiss.PCAMatrix(d_orig, self.pca_dim)
            pca.train(normalized_embeddings[:self.num_train_pca])
            assert pca.is_trained
            transformed_feats = pca.apply_py(normalized_embeddings)

            # Cluster transformed feats with kmeans
            results = []
            gt = torch.cat(self.gt, dim=0).cpu().numpy()[valid_masks]
            for k in self.num_pred_classes:  # [500, 300, 21]
                kmeans = faiss.Kmeans(self.pca_dim, k, niter=50, nredo=5, seed=1, verbose=True, gpu=False,
                                      spherical=False)
                kmeans.train(transformed_feats)
                _, pred_labels = kmeans.index.search(transformed_feats, 1)
                clusters = pred_labels.squeeze()

                # Filter predictions by valid masks (removes voc boundary gt class)
                pred_flattened = clusters.reshape(valid_masks.shape[0], 1, res_w, res_w)[valid_masks]
                assert len(np.unique(pred_flattened)) == k
                assert np.max(pred_flattened) == k - 1

                # Calculate mIoU. Do many-to-one matching if k > self.num_gt_classes.
                if k == self.num_gt_classes:
                    results.append((k, k, self.compute_miou(gt, pred_flattened, k, self.num_gt_classes,
                                                            many_to_one=False)))
                else:
                    results.append((k, k, self.compute_miou(gt, pred_flattened, k, self.num_gt_classes,
                                                            many_to_one=True)))
                    results.append((k, f"{k}_prec", self.compute_miou(gt, pred_flattened, k, self.num_gt_classes,
                                                                      many_to_one=True, precision_based=True)))
            return results


def eval_jac(gt: torch.Tensor, pred_mask: torch.Tensor, with_boundary: bool = True) -> float:
    """
    Calculate Intersection over Union averaged over all pictures. with_boundary flag, if set, doesn't filter out the
    boundary class as background.
    """
    jacs = 0
    for k, mask in enumerate(gt):
        if with_boundary:
            gt_fg_mask = (mask != 0).float()
        else:
            gt_fg_mask = ((mask != 0) & (mask != 255)).float()
        intersection = gt_fg_mask * pred_mask[k]
        intersection = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
        union = (gt_fg_mask + pred_mask[k]) > 0
        union = torch.sum(torch.sum(union, dim=-1), dim=-1)
        jacs += intersection / union
    res = jacs / gt.size(0)
    print(res)
    return res.item()


def process_attentions(attentions: torch.Tensor, spatial_res: int, threshold: float = 0.6, blur_sigma: float = 0.6) \
        -> torch.Tensor:
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask] = 0
    return th_attn.detach()


def neq_load_customized(model, pretrained_dict):
    """
    load pre-trained model in a non-equal way,
    when new model has been partially modified
    """
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)

    print('\n-----------------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print('===================================\n')

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def neq_load_external(model, pretrained_dict):
    """
    load pre-trained model from external source
    """
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k.startswith('model'):
            k = k.removeprefix('model.') # for Leopart
        if 'backbone.' + k in model_dict:
            tmp['backbone.' + k] = v
        else:
            print(k)

    print('\n-----------------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in tmp:
            print(k)
    print('===================================\n')

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()


def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def calc_accuracy(output, target):
    '''output: (B, N); target: (B)'''
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


def calc_accuracy_binary(output, target):
    '''output, target: (B, N), output is logits, before sigmoid '''
    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    del pred, output, target
    return acc


def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean) == len(std) == 3
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


class AccuracyTable(object):
    '''compute accuracy for each class'''

    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {'count': 0, 'correct': 0}
            self.dict[j]['count'] += 1
            if i == j:
                self.dict[j]['correct'] += 1

    def print_table(self, label):
        for key in self.dict.keys():
            acc = self.dict[key]['correct'] / self.dict[key]['count']
            print('%s: %2d, accuracy: %3d/%3d = %0.6f' \
                  % (label, key, self.dict[key]['correct'], self.dict[key]['count'], acc))


class ConfusionMeter(object):
    '''compute and show confusion matrix'''

    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        pred, tar = pred.cpu().numpy(), tar.cpu().numpy()
        pred = np.squeeze(pred)
        tar = np.squeeze(tar)
        for p, t in zip(pred.flat, tar.flat):
            self.mat[p][t] += 1

    def print_mat(self):
        print('Confusion Matrix: (target in columns)')
        print(self.mat)

    def plot_mat(self, path, dictionary=None, annotate=False):
        plt.figure(dpi=600)
        plt.imshow(self.mat,
                   cmap=plt.cm.jet,
                   interpolation=None,
                   extent=(0.5, np.shape(self.mat)[0] + 0.5, np.shape(self.mat)[1] + 0.5, 0.5))
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    plt.annotate(str(int(self.mat[x][y])), xy=(y + 1, x + 1),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=8)

        if dictionary is not None:
            plt.xticks([i + 1 for i in range(width)],
                       [dictionary[i] for i in range(width)],
                       rotation='vertical')
            plt.yticks([i + 1 for i in range(height)],
                       [dictionary[i] for i in range(height)])
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, format='svg')
        plt.clf()

        # for i in range(width):
        #     if np.sum(self.mat[i,:]) != 0:
        #         self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
        #     if np.sum(self.mat[:,i]) != 0:
        #         self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        # print('Average Precision: %0.4f' % np.mean(self.precision))
        # print('Average Recall: %0.4f' % np.mean(self.recall))


if __name__ == '__main__':
    pass
