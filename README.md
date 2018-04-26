# "Zero-Shot" Super-Resolution using Deep Internal Learning  
Official implementation

Paper: https://arxiv.org/abs/1712.06087  
Project page: http://www.wisdom.weizmann.ac.il/~vision/zssr/

----------

# Usage:

Quick usage:  

```
python run_ZSSR.py
```
(First, put your desired low-res files in **<ZSSR_path>/test_data/**. 
Results will be generated to **<ZSSR_path>/results/<name_date>/**.)

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
Run on a specific GPU:
```
python run_ZSSR.py <config> 0
```
Run multiple files efficiently on multiple GPUs
```
python run_ZSSR.py <config> all
```
