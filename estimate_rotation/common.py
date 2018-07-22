from enum import Enum, auto, unique


@unique
class AngleEncoding(Enum):
    """
    Discussion about why it is better to encode angle as sin/cos rather than directly:
    http://practicalcryptography.com/miscellaneous/machine-learning/encoding-variables-neural-networks/#angles
    https://stats.stackexchange.com/questions/218407/encoding-angle-data-for-neural-network
    """
    # angle in [-180, 180)
    DEGREES = auto()
    # angle in [-pi, pi)
    RADIANS = auto()
    # value in [-1, 1) representing angle in radians scaled down pi times; note, however, that gradients (i.e.,
    # differences) would not be uniformly distributed in the same interval. Take a look at this book, page 17, to get
    # the pdf of the gradients, compute its mean and variance and adjust weight init for the last layer, accordingly.
    # https://www.inf.ed.ac.uk/publications/thesis/online/IM090722.pdf
    # Note, again, that most schemes for weight init also activations to have mean=0 and std=1, so, maybe find a
    # compromise between the two constraints (this distribution does not have std=1, either, as it is uniform).
    UNIT = auto()
    # represent target angle as a (sin, cos) tuple; during prediction apply atan2(output2, output1) to the network's
    # outputs to get the angle (note that network outputs don't have to be bounded to [-1, 1], so a better name for them
    # would be x, y instead of sin, cos)
    SINCOS = auto()
    # target is a one-hot vector of probabilities, 1 marks the sub-interval of [-pi, pi) that contains the angle;
    CLASSES = auto()
