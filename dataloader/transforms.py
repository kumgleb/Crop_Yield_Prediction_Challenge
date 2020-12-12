from torchvision.transforms import transforms


class Normalize(object):
    def __init__(self, mean, std):
        self.normalizer = transforms.Normalize(mean, std)

    def __call__(self, sample):
        s2_bands = sample['s2_bands']
        sample['s2_bands'] = self.normalizer(s2_bands)
        return sample


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, list))
        self.resizer = transforms.Resize(output_size)

    def __call__(self, sample):
        s2_bands = sample['s2_bands']
        sample['s2_bands'] = self.resizer(s2_bands)
        return sample
