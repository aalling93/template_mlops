

class ImageCompose(object):
    """ "
    A compose class that will apply a list of transforms to an image.
    This is only for the iamges. kinda pre processing.

    For augmentation, use the augmentation class.

    Args:
        transforms (list): list of transforms to apply.
            Transforms can be found in the transorms module.

    Example:
        >>> transforms = ImageCompose([
        >>>     transforms.RandomHorizontalFlip(),
        >>>     transforms.RandomVerticalFlip(),
        >>>     transforms.RandomRotate90(),
        >>>     ])


    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img