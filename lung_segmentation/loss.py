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
