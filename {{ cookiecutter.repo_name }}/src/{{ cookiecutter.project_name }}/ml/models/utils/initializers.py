import torch.nn as nn
import torch
import math


def custom_init(m):
    """
    Custom weight and bias initialization for a detection head with mixed activation functions.

    This will add different initializations for differetn detection ehad depending on activation function etc..


    The initializer applies Kaiming initialization for weights in layers with ReLU or SiLU activations,
    and Xavier initialization for weights in layers with Tanh activations. The bias in all layers is
    initialized based on the prior probability of object presence.

    Kaiming Initialization:
        Named after the first author of the paper that introduced it (He et al., 2015), Kaiming
        initialization is designed for layers with ReLU activations. It initializes the weights of
        the layer with values that take into account the size of the previous layer, which can lead
        to more effective training when using ReLU activations.

    Xavier Initialization:
        Named after the first author of the paper that introduced it (Glorot and Bengio, 2010),
        Xavier initialization is designed to keep the scale of the gradients roughly the same in all
        layers. It's particularly effective when used with sigmoid or tanh activation functions.

    Bias Initialization:
        The bias is initialized with the value -log((1 - prior_prob) / prior_prob), where prior_prob
        is the prior probability that an object is present. This initialization helps to start the
        training process with a reasonable guess: that objects are rare. The mathematical reasoning
        behind this is that we want the initial predicted probability of object presence to be
        close to the prior probability. Given that the sigmoid function is used to predict this
        probability, setting the bias to -log((1 - prior_prob) / prior_prob) ensures that the
        initial predicted probability is close to prior_prob.

        from the invers sigmoid
        \sigma(x) = \frac{1}{1 + e^{-x}}
        we can find the log-odds..
        \sigma^{-1}(p) = -\log\left(\frac{1 - p}{p}\right)
        Here, p is the probability of the positive class (i.e., an object being present). If we set p to be our prior_prob, we get the expression -math.log((1 - prior_prob) / prior_prob).

        The reason for using this expression for bias initialization is that we want the initial predicted probability of object presence to be close to the prior probability.
        Given that the sigmoid function is used to predict this probability, setting the bias to -math.log((1 - prior_prob) / prior_prob) ensures that the initial predicted probability is close to prior_prob.

        In the case of prior_prob = 0.01, the initial predicted probability will be close to 0.01,
        which is a reasonable starting point because we typically expect that objects are rare. We have very few ship in the image.

        if we set p = 0.01, we get bias of -4.59...

        The negative bias initialization is a common technique used in object detection models, especially for the final layer of the detection head that predicts the probability of object presence.

        TODO: removed the sigmoid and get raw digts. Therefore removing this part.


    For nn.Conv2d or nn.Linear layers:
    If the activation function is nn.ReLU or nn.SiLU, the weights are initialized using the Kaiming normal initializer with nonlinearity='relu'.
    If the activation function is nn.Tanh, the weights are initialized using the Xavier uniform initializer.
    For other activation functions, the weights are initialized using the Kaiming normal initializer with nonlinearity='linear'.
    If the layer has a bias, it's initialized to 0.0001.

    For nn.BatchNorm2d layers:
    The weights are initialized to 1.
    The bias and running mean are initialized to 0.
    The running variance is initialized to 1 - epsilon, where epsilon is a small constant to prevent division by zero.

    Args:
        m (nn.Module): The module to be initialized.

    """
    prior_prob = 0.001
    epsilon = 1e-5

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if hasattr(m, "activation"):
            if isinstance(m.activation, nn.ReLU) or isinstance(m.activation, nn.SiLU):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m.activation, nn.Tanh):
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        else:
            # Default initialization if no activation attribute
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

        if m.bias is not None:
            m.bias.data.fill_(prior_prob)
            # m.bias.data.fill_(-math.log((1 - prior_prob) / prior_prob))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        m.running_mean.fill_(0.001)
        m.running_var.fill_(1 - epsilon)


def init_linear_weights(m):
    """
    Custom weight initialization for linear (fully connected) layers.

    This initializer uses the Xavier (Glorot) initialization method, which is designed to keep the scale of
    the gradients roughly the same in all layers. In the case of Xavier initialization, the weights are sampled
    from a uniform distribution that's centered on 0 and has a variance of Var(W) = 1/N, where N is the number
    of input neurons.

    The biases are initialized with a constant value of 0.01 to slightly break symmetry and prevent zero gradients.

    Xavier initialization is generally good for situations where you want to avoid vanishing and exploding gradients,
    especially in deep networks.

    Args:
        m (nn.Module): Linear layer module.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_conv_weights(m):
    """
    Custom weight initialization for convolutional layers.

    This initializer uses the Kaiming (He) initialization method, which is a method designed to keep the scale of
    the gradients roughly the same in all layers. In the case of Kaiming initialization, the weights are sampled
    from a normal distribution that's centered on 0 and has a variance of Var(W) = 2/N, where N is the number of
    input neurons. This initialization is particularly beneficial for layers with ReLU activations.

    The biases are initialized with a constant value of 0.01 to slightly break symmetry and prevent zero gradients.

    Kaiming initialization is generally good for situations where you want to avoid vanishing and exploding gradients,
    especially in deep networks with ReLU activations.

    Args:
        m (nn.Module): Convolutional layer module.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0.00001)


def init_detection_head(m):
    """
    Custom weight initialization for the detection head.

    This initializer uses a normal distribution to initialize the weights of the detection head. The weights are
    sampled from a normal distribution that's centered on 0 and has a standard deviation of 0.01. This helps to
    ensure that the weights are small and the network does not suffer from exploding gradients.

    The biases are initialized with a constant value of 0.01 to slightly break symmetry and prevent zero gradients.

    This initialization is particularly beneficial for detection heads, where it's important to start with smaller
    weights to prevent the gradients from exploding, especially during the early stages of training.

    Args:
        m (nn.Module): Detection head module.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0, std=0.01)  # Initialize with smaller weights
        if m.bias is not None:
            m.bias.data.fill_(0.00001)


# Optionally, you can fine-tune specific layers within the detection head for different object scales
def fine_tune_head_for_small_objects(m):
    """
    Custom weight initialization for convolutional layers in the detection head that are responsible for detecting small objects.

    This initializer uses a normal distribution to initialize the weights of the convolutional layers with a kernel size of 3x3 or 1x1 and 256 input channels.
    The weights are sampled from a normal distribution that's centered on 0 and has a standard deviation of 0.001. This helps to ensure that the weights are small and the network does not suffer from exploding gradients.

    This initialization is particularly beneficial for layers responsible for detecting small objects,
    where it's important to start with smaller weights to prevent the gradients from exploding, especially during the early stages of training.

    Args:
        m (nn.Module): Convolutional layer module.
    """
    if isinstance(m, nn.Conv2d) and m.in_channels == 256:
        # Fine-tune convolution layers with kernel size 3x3 or 1x1
        if m.kernel_size == (3, 3) or m.kernel_size == (1, 1):
            nn.init.normal_(m.weight, mean=0, std=0.00001)  # Initialize with smaller weights


def init_selu_weights(m):
    """
    Custom weight initialization for layers with SELU activation function.

    This initializer uses the LeCun initialization method, which is designed to maintain the mean and variance of the inputs constant across layers.
    In the case of LeCun initialization, the weights are sampled from a normal distribution that's centered on 0 and has a standard deviation of sqrt(1/N), where N is the number of input neurons.

    LeCun initialization is particularly beneficial for layers with SELU activations, as it helps to ensure the self-normalizing property of the SELU activation function.

    Args:
        m (nn.Module): Layer module.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def initialize_biases(m, prior_prob: float = 0.0001):
    """
    Custom bias initialization for convolutional layers in the detection head.

    This initializer uses a prior probability to set the initial bias terms, which can help in faster convergence during training.
    The bias terms are initialized with the negative logarithm of the ratio (1 - prior_prob) / prior_prob.

    This initialization is particularly beneficial for the detection head, where setting the initial bias terms with a prior probability can help in faster convergence and better detection performance.

    Args:
        m (nn.Module): Convolutional layer module.
        prior_prob (float): Prior probability used for bias initialization.
    """
    if isinstance(m, nn.Conv2d) and m.bias is not None:
        b = m.bias.view(1, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
