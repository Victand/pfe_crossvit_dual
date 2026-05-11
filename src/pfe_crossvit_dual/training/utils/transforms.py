import torchvision.transforms.functional as F


def apply_colorjitter(img, fn_idx, brightness, contrast, saturation, hue):
    for fn_id in fn_idx:
        if fn_id == 0 and brightness is not None:
            img = F.adjust_brightness(img, brightness)
        elif fn_id == 1 and contrast is not None:
            img = F.adjust_contrast(img, contrast)
        elif fn_id == 2 and saturation is not None:
            img = F.adjust_saturation(img, saturation)
        elif fn_id == 3 and hue is not None:
            img = F.adjust_hue(img, hue)
    return img


def scale_translation(translations, old_size, new_size):
    scale_y = new_size[-2] / old_size[-2]
    scale_x = new_size[-1] / old_size[-1]
    new_translations = (int(translations[0] * scale_x), int(translations[1] * scale_y))
    return new_translations


def scale_crop_params(i, j, h, w, src_size, tgt_size):
    src_h, src_w = src_size
    tgt_h, tgt_w = tgt_size

    scale_y = tgt_h / src_h
    scale_x = tgt_w / src_w

    i_t = int(i * scale_y)
    j_t = int(j * scale_x)
    h_t = int(h * scale_y)
    w_t = int(w * scale_x)

    return i_t, j_t, h_t, w_t
