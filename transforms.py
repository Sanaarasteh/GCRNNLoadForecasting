import torch


class ToTensor(object):
    """
    This class implements a callable class which converts Numpy samples to Tensor samples
    """
    def __call__(self, sample):
        """
        :param sample: A dictionary of ndarray objects
        :return: A dictionary tensor objects
        """

        x, adj, y = sample['x'], sample['adj'], sample['y']

        new_sample = {'x': torch.tensor(x, dtype=torch.float32),
                      'adj': torch.tensor(adj, dtype=torch.float32),
                      'y': torch.tensor(y, dtype=torch.float32)}

        return new_sample


class SimpleToTensor(object):
    """
    This class implements a callable class which converts Numpy samples to Tensor samples
    """
    def __call__(self, sample):
        """
        :param sample: A dictionary of ndarray objects
        :return: A dictionary tensor objects
        """

        x, y = sample['x'], sample['y']

        new_sample = {'x': torch.tensor(x, dtype=torch.float32), 'y': torch.tensor(y, dtype=torch.float32)}

        return new_sample
