import os


class Config:
    # network meta params
    python_path = '/home/assafsho/PycharmProjects/network/venv/bin/python2.7'
    scale_factors = [[2.0, 2.0]]  # list of pairs (vertical, horizontal) for gradual increments in resolution
    base_change_sfs = []  # list of scales after which the input is changed to be the output (recommended for high sfs)
    max_iters = 3000
    min_iters = 256
    min_learning_rate = 9e-6  # this tells the algorithm when to stop (specify lower than the last learning-rate)
    width = 64
    depth = 8
    output_flip = True  # geometric self-ensemble (see paper)
    downscale_method = 'cubic'  # a string ('cubic', 'linear'...), has no meaning if kernel given
    upscale_method = 'cubic'  # this is the base interpolation from which we learn the residual (same options as above)
    downscale_gt_method = 'cubic'  # when ground-truth given and intermediate scales tested, we shrink gt to wanted size
    learn_residual = True  # when true, we only learn the residual from base interpolation
    init_variance = 0.1  # variance of weight initializations, typically smaller when residual learning is on
    back_projection_iters = [10]  # for each scale num of bp iterations (same length as scale_factors)
    random_crop = True
    crop_size = 128
    noise_std = 0.0  # adding noise to lr-sons. small for real images, bigger for noisy images and zero for ideal case
    init_net_for_each_sf = False  # for gradual sr- should we optimize from the last sf or initialize each time?

    # Params concerning learning rate policy
    learning_rate = 0.001
    learning_rate_change_ratio = 1.5  # ratio between STD and slope of linear fit, under which lr is reduced
    learning_rate_policy_check_every = 60
    learning_rate_slope_range = 256

    # Data augmentation related params
    augment_leave_as_is_probability = 0.05
    augment_no_interpolate_probability = 0.45
    augment_min_scale = 0.5
    augment_scale_diff_sigma = 0.25
    augment_shear_sigma = 0.1
    augment_allow_rotation = True  # recommended false for non-symmetric kernels

    # params related to test and display
    run_test = True
    run_test_every = 50
    display_every = 20
    name = 'test'
    plot_losses = False
    result_path = os.path.dirname(__file__) + '/results'
    create_results_dir = True
    input_path = local_dir = os.path.dirname(__file__) + '/test_data'
    create_code_copy = True  # save a copy of the code in the results folder to easily match code changes to results
    display_test_results = True
    save_results = True

    def __init__(self):
        # network meta params that by default are determined (by other params) by other params but can be changed
        self.filter_shape = ([[3, 3, 3, self.width]] +
                             [[3, 3, self.width, self.width]] * (self.depth-2) +
                             [[3, 3, self.width, 3]])


########################################
# Some pre-made useful example configs #
########################################

# Basic default config (same as not specifying), non-gradual SRx2 with default bicubic kernel (Ideal case)
# example is set to run on set14
X2_ONE_JUMP_IDEAL_CONF = Config()
X2_ONE_JUMP_IDEAL_CONF.input_path = os.path.dirname(__file__) + '/set14'

# Same as above but with visualization (Recommended for one image, interactive mode, for debugging)
X2_IDEAL_WITH_PLOT_CONF = Config()
X2_IDEAL_WITH_PLOT_CONF.plot_losses = True
X2_IDEAL_WITH_PLOT_CONF.run_test_every = 20
X2_IDEAL_WITH_PLOT_CONF.input_path = os.path.dirname(__file__) + '/example_with_gt'

# Gradual SRx2, to achieve superior results in the ideal case
X2_GRADUAL_IDEAL_CONF = Config()
X2_GRADUAL_IDEAL_CONF.scale_factors = [[1.0, 1.5], [1.5, 1.0], [1.5, 1.5], [1.5, 2.0], [2.0, 1.5], [2.0, 2.0]]
X2_GRADUAL_IDEAL_CONF.back_projection_iters = [6, 6, 8, 10, 10, 12]
X2_GRADUAL_IDEAL_CONF.input_path = os.path.dirname(__file__) + '/set14'

# Applying a given kernel. Rotations are canceled sense kernel may be non-symmetric
X2_GIVEN_KERNEL_CONF = Config()
X2_GIVEN_KERNEL_CONF.output_flip = False
X2_GIVEN_KERNEL_CONF.augment_allow_rotation = False
X2_GIVEN_KERNEL_CONF.back_projection_iters = [2]
X2_GIVEN_KERNEL_CONF.input_path = os.path.dirname(__file__) + '/kernel_example'

# An example for a typical setup for real images. (Kernel needed + mild unknown noise)
# back-projection is not recommended because of the noise.
X2_REAL_CONF = Config()
X2_REAL_CONF.output_flip = False
X2_REAL_CONF.back_projection_iters = [0]
X2_REAL_CONF.input_path = os.path.dirname(__file__) + '/real_example'
X2_REAL_CONF.noise_std = 0.0125
X2_REAL_CONF.augment_allow_rotation = False
X2_REAL_CONF.augment_scale_diff_sigma = 0
X2_REAL_CONF.augment_shear_sigma = 0
X2_REAL_CONF.augment_min_scale = 0.75
