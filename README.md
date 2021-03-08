# vkJAX
[JAX](https://github.com/google/jax) interpreter based on [Vulkan Kompute](https://github.com/EthicalML/vulkan-kompute)


***
### Minimal Example
```python
import numpy as np, jax.numpy as jnp
import vkjax

def jax_fun(x,W,b):
  return jnp.dot(x, W) + b

vkfun = vkjax.wrap(jax_fun)

#this runs on the GPU, powered by vulkan
y = vkfun(
    np.random.random([8,128]),
    np.random.random([128,16]),
    np.random.random([16])
)
```


***
### Integration with [Elegy](https://github.com/poets-ai/elegy)
`pip install elegy==0.7.1`
```python
import elegy
from vkjax.elegy import vkModel
import PIL.Image, urllib, numpy as np

#auto-download a pretrained ResNet50
r50     = elegy.nets.ResNet50(weights='imagenet')
vkmodel = vkModel(r50)

#download an example image
fname,_ = urllib.request.urlretrieve('https://upload.wikimedia.org/wikipedia/commons/e/e4/A_French_Bulldog.jpg')
image   = np.array(PIL.Image.open(fname).resize([224,224])) / np.float32(255)

#run inference on the GPU, powered by vulkan
y = vkmodel.predict(image[np.newaxis])
assert y.argmax() == 245  #ImageNet label #245: French Bulldog
```


***
### Current Limitations
- Only an **incomplete** subset of all JAX/XLA primitives is implemented. Feel free to create a new issue, if you encounter a `NotImplementedError`.
- The performance might be **slow**, even slower than JAX' (very optimized) CPU backend. The current development focus lies on compatibility. Speed optimizations will follow.


