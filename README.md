# "Zero-Shot" Super-Resolution using Deep Internal Learning  
Official implementation

Paper: https://arxiv.org/abs/1712.06087  
Project page: http://www.wisdom.weizmann.ac.il/~vision/zssr/

----------

# Usage:

## Quick usage on your data:  
(First, put your desired low-res files in ```**<ZSSR_path>/test_data/**```.  
Results will be generated to ```**<ZSSR_path>/results/<name_date>/**```.  
data must be *.png type)
```
python run_ZSSR.py
```

## General usage:
```
python run_ZSSR.py <config> <gpu-optional>
```
While ``` <config> ``` is an instance of configs.Config class (at configs.py) or 0 for default configuration.  
and ``` <gpu> ``` is an optional parameter to determine how to use available GPUs (see next section).

For using given kernels, you must have a kernels for each input file and each scale-factor named as follows:  
``` <inpu_file_name>_<scale_factor_ind_starting_0>.mat ```  
Kernels are MATLAB files containing a matrix named "Kernel".  

If gound-truth exists and true-error monitoring is wanted, then ground truth should be named as follows:  
``` <inpu_file_name>_gt.png ```  


## GPU options
Run on a specific GPU:
```
python run_ZSSR.py <config> 0
```
Run multiple files efficiently on multiple GPUs.  
**Before using this option make sure you update in the configs.py file the ***python_path*** parameter**
```
python run_ZSSR.py <config> all
```

## Quick usage examples (applied on provided data examples):  
Usage example to test 'Set14', Gradual SR (~0.3dB better results, 6x Runtime)
```
python run_ZSSR.py X2_GRADUAL_IDEAL_CONF
```
Usage example to test 'Set14' (Non-Gradual SR)
```
python run_ZSSR.py X2_ONE_JUMP_IDEAL_CONF
```
Visualization while running (Recommended for one image, interactive mode, for debugging)
```
python run_ZSSR.py X2_ONE_JUMP_IDEAL_CONF
```
Applying a given kernel
```
python run_ZSSR.py X2_GIVEN_KERNEL_CONF
```
Run on a real image
```
X2_REAL_CONF
```

