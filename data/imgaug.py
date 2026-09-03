import cv2
import torchvision.transforms as tfs


def Common(image):

    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def Aug(image):
    try:
        affine = tfs.RandomAffine(
            degrees=(-15, 15),
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=128,
        )
    except TypeError:
        affine = tfs.RandomAffine(
            degrees=(-15, 15),
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fillcolor=128,
        )

    img_aug = tfs.Compose([affine])
    image = img_aug(image)

    return image


def GetTransforms(image, target=None, type='common'):
    if target is not None:
        raise Exception('Target is not support now  ')
    if type.strip() == 'Common':
        image = Common(image)
        return image
    elif type.strip() == 'None':
        return image
    elif type.strip() == 'Aug':
        image = Aug(image)
        return image
    else:
        raise Exception(
            'Unknown transforms_type : '.format(type))

