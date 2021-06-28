
import torch
import numpy as np

from utils.confusionmatrix import ConfusionMatrix


class IoU(object):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.
        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return conf_matrix, iou, np.nanmean(iou)




























# import torch
# import numpy as np
# from torch.autograd import Function

# device = torch.device('cuda:0')
# class DiceCoeff(Function):
#     """Dice coeff for individual examples"""

#     def forward(self, input, target):
#         eps = 0.0001
#         input = (input > 0.5).float()
# #         print(input.shape)
#         intersection = torch.sum(2*torch.sum(torch.mul(input,target),2),1)
#         union = torch.sum(torch.sum(input,2),1) + torch.sum(torch.sum(target,2),1)

#         dice = torch.mean(torch.divide(intersection + eps, union+eps))
#         return dice

#     # This function has only a single output, so it gets only one gradient
#     def backward(self, grad_output):

#         input, target = self.saved_variables
#         grad_input = grad_target = None

#         if self.needs_input_grad[0]:
#             grad_input = grad_output * 2 * (target * self.union - self.inter) \
#                          / (self.union * self.union)
#         if self.needs_input_grad[1]:
#             grad_target = None

#         return grad_input, grad_target


# def dice_coeff(input, target):
#     """Dice coeff for batches"""
#     if input.is_cuda:
#         s = torch.FloatTensor(1).to(device).zero_()
#     else:
#         s = torch.FloatTensor(1).zero_()

#     for i, c in enumerate(zip(input, target)):
#         s = s + DiceCoeff().forward(c[0], c[1])

#     return s / (i + 1)



# def dice_coef(y_true, y_pred):
    
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = np.sum(y_true_f * y_pred_f)
#     smooth = 0.0001
#     return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# def dice_coef_multilabel(y_true, y_pred, numLabels):
#     dice=0
#     for index in range(numLabels):
#         dice += dice_coef(y_true[:,index, :,:], y_pred[:,index,:,:])
#     return dice/numLabels # taking average