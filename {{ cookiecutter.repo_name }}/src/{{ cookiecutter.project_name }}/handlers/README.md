# Profilers

Profilers are essential tools for analyzing the performance of code. They provide insights into various aspects of program execution, such as:

- **Function call frequency**: Determining how many times each function is invoked.
- **Execution time**: Measuring the time spent in each function, both individually and cumulatively.

By leveraging profilers, developers can identify performance bottlenecks and optimize code efficiency.


Profiling assists in:

- **Identifying inefficient code segments**: Highlighting parts of the code that consume excessive time or resources.
- **Optimizing performance**: Guiding targeted improvements to enhance execution speed.
- **Understanding code behavior**: Providing a clear picture of how code executes, which is crucial for debugging and optimization.


Various profilers are available, including:

- **cProfile**: Python's built-in deterministic profiler, implemented in C, which offers detailed statistics with minimal overhead.
- **Pyinstrument**: A statistical profiler that provides an overview of time spent in different parts of the code.
- **Perf**: A tool for measuring and analyzing performance, often used for benchmarking.

In machine learning, frameworks like PyTorch offer specialized profilers tailored to their operations.


The `cProfile` module is a deterministic profiler that helps answer questions such as:

- How many times was a particular function called?
- How much total time was spent inside that function?

To profile a function using `cProfile`:

```python
import cProfile

def my_function():
    # Your code here
    pass

cProfile.run('my_function()')


```

Pytorch profiling is a bit more difficult. Here, we are mixing CPU and GPU. Specifially, we  looking for bottlenecks, e.g. where we are loading data from CPU to GPU or whatever.


```python
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)


with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    model(inputs)


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     aten::conv2d         0.20%      46.517us        74.69%      17.208ms     860.395us            20  
                aten::convolution         0.50%     115.194us        74.49%      17.161ms     858.069us            20  
               aten::_convolution         0.29%      67.945us        73.99%      17.046ms     852.309us            20  
         aten::mkldnn_convolution        73.26%      16.878ms        73.70%      16.978ms     848.912us            20  
                 aten::batch_norm         0.11%      24.207us        14.08%       3.243ms     162.157us            20  
     aten::_batch_norm_impl_index         0.19%      43.890us        13.97%       3.219ms     160.947us            20  
          aten::native_batch_norm        13.37%       3.079ms        13.77%       3.171ms     158.567us            20  
                 aten::max_pool2d         0.02%       3.697us         7.28%       1.678ms       1.678ms             1  
    aten::max_pool2d_with_indices         7.27%       1.674ms         7.27%       1.674ms       1.674ms             1  
                       aten::add_         1.43%     328.701us         1.43%     328.701us      11.739us            28  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 23.038ms

```

and

```python


print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------------------------------------------------------------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls                                                                      Input Shapes  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------------------------------------------------------------------  
                     aten::conv2d         0.07%      15.579us        18.98%       4.372ms       1.093ms             4                             [[5, 64, 56, 56], [64, 64, 3, 3], [], [], [], [], []]  
                aten::convolution         0.13%      29.365us        18.91%       4.356ms       1.089ms             4                     [[5, 64, 56, 56], [64, 64, 3, 3], [], [], [], [], [], [], []]  
               aten::_convolution         0.10%      22.170us        18.78%       4.327ms       1.082ms             4     [[5, 64, 56, 56], [64, 64, 3, 3], [], [], [], [], [], [], [], [], [], [], []]  
         aten::mkldnn_convolution        18.56%       4.276ms        18.69%       4.305ms       1.076ms             4                             [[5, 64, 56, 56], [64, 64, 3, 3], [], [], [], [], []]  
                     aten::conv2d         0.02%       3.698us        12.47%       2.872ms     957.275us             3                            [[5, 512, 7, 7], [512, 512, 3, 3], [], [], [], [], []]  
                aten::convolution         0.04%       9.377us        12.45%       2.868ms     956.042us             3                    [[5, 512, 7, 7], [512, 512, 3, 3], [], [], [], [], [], [], []]  
               aten::_convolution         0.02%       5.680us        12.41%       2.859ms     952.916us             3    [[5, 512, 7, 7], [512, 512, 3, 3], [], [], [], [], [], [], [], [], [], [], []]  
         aten::mkldnn_convolution        12.36%       2.848ms        12.38%       2.853ms     951.023us             3                            [[5, 512, 7, 7], [512, 512, 3, 3], [], [], [], [], []]  
                     aten::conv2d         0.03%       5.840us         9.77%       2.252ms     750.615us             3                          [[5, 128, 28, 28], [128, 128, 3, 3], [], [], [], [], []]  
                aten::convolution         0.06%      14.396us         9.75%       2.246ms     748.668us             3                  [[5, 128, 28, 28], [128, 128, 3, 3], [], [], [], [], [], [], []]  
               aten::_convolution         0.04%       8.866us         9.69%       2.232ms     743.869us             3  [[5, 128, 28, 28], [128, 128, 3, 3], [], [], [], [], [], [], [], [], [], [], []]  
         aten::mkldnn_convolution         9.59%       2.209ms         9.65%       2.223ms     740.914us             3                          [[5, 128, 28, 28], [128, 128, 3, 3], [], [], [], [], []]  
                     aten::conv2d         0.04%       8.746us         9.22%       2.124ms       2.124ms             1                             [[5, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], []]  
                aten::convolution         0.13%      28.954us         9.18%       2.115ms       2.115ms             1                     [[5, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], [], [], []]  
               aten::_convolution         0.06%      12.905us         9.06%       2.086ms       2.086ms             1     [[5, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], [], [], [], [], [], [], []]  
         aten::mkldnn_convolution         8.87%       2.045ms         9.00%       2.073ms       2.073ms             1                             [[5, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], []]  
                     aten::conv2d         0.02%       3.877us         8.95%       2.062ms     687.390us             3                          [[5, 256, 14, 14], [256, 256, 3, 3], [], [], [], [], []]  
                aten::convolution         0.04%       8.104us         8.93%       2.058ms     686.098us             3                  [[5, 256, 14, 14], [256, 256, 3, 3], [], [], [], [], [], [], []]  
               aten::_convolution         0.02%       4.960us         8.90%       2.050ms     683.397us             3  [[5, 256, 14, 14], [256, 256, 3, 3], [], [], [], [], [], [], [], [], [], [], []]  
         aten::mkldnn_convolution         8.85%       2.040ms         8.88%       2.045ms     681.743us             3                          [[5, 256, 14, 14], [256, 256, 3, 3], [], [], [], [], []]  
                 aten::max_pool2d         0.02%       3.697us         7.28%       1.678ms       1.678ms             1                                           [[5, 64, 112, 112], [], [], [], [], []]  
    aten::max_pool2d_with_indices         7.27%       1.674ms         7.27%       1.674ms       1.674ms             1                                           [[5, 64, 112, 112], [], [], [], [], []]  
                 aten::batch_norm         0.01%       3.236us         5.45%       1.256ms       1.256ms             1                       [[5, 64, 112, 112], [64], [64], [64], [64], [], [], [], []]  
     aten::_batch_norm_impl_index         0.03%       6.432us         5.44%       1.253ms       1.253ms             1                       [[5, 64, 112, 112], [64], [64], [64], [64], [], [], [], []]  
          aten::native_batch_norm         5.34%       1.230ms         5.41%       1.246ms       1.246ms             1                           [[5, 64, 112, 112], [64], [64], [64], [64], [], [], []]  
                     aten::conv2d         0.01%       1.553us         5.12%       1.180ms       1.180ms             1                            [[5, 64, 56, 56], [128, 64, 3, 3], [], [], [], [], []]  
                aten::convolution         0.03%       6.543us         5.12%       1.178ms       1.178ms             1                    [[5, 64, 56, 56], [128, 64, 3, 3], [], [], [], [], [], [], []]  
               aten::_convolution         0.01%       3.356us         5.09%       1.172ms       1.172ms             1    [[5, 64, 56, 56], [128, 64, 3, 3], [], [], [], [], [], [], [], [], [], [], []]  
         aten::mkldnn_convolution         5.05%       1.163ms         5.07%       1.169ms       1.169ms             1                            [[5, 64, 56, 56], [128, 64, 3, 3], [], [], [], [], []]  
                 aten::batch_norm         0.03%       6.713us         3.80%     876.427us     219.107us             4                         [[5, 64, 56, 56], [64], [64], [64], [64], [], [], [], []]  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------------------------------------------------------------------  
Self CPU time total: 23.038ms



# or

prof = profile(activities=[ProfilerActivity.CPU], record_shapes=True)
prof.start()

model(inputs)

prof.stop()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
```



