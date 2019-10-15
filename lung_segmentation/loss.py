"Losses to be used for CNN training"
from keras import backend as K


def dice_coefficient(y_true, y_pred):
    "Dice coefficient loss"
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2.0 * intersection + smoothing_factor)
            /(K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor))


def loss_dice_coefficient_error(y_true, y_pred):
    "Dice coefficient"
    return -dice_coefficient(y_true, y_pred)


def jaccard_distance(y_true, y_pred, smooth=100):
    "Jaccard distance"
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    "Jaccard distance loss"
    return (1 - jaccard_distance(y_true, y_pred)) * smooth
