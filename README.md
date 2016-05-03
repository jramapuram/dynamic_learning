# Dynamic Learning
A try to build a secondary neural network to emit a scalar to zero out the loss.
The hope is that this will cause the network to be able to learn forever (i.e. use SGD at all times) since the gradient will be zero.
This removes the need for a difference between a train and a test step

## Usage
```python
python dynamic_learning.py
```
