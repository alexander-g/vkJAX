import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import elegy
import vkjax.elegy as vkelegy




class Module0(elegy.Module):
    def call(self, x):
        x = elegy.nn.Linear(77)(x)
        return x



def test_basic_predict0():
    x       = np.random.random([1,224,224,3])

    r18     = elegy.nets.ResNet18(weights='imagenet')

    model   = elegy.Model(r18)
    vkmodel = vkelegy.vkModel(r18)
    
    ypred   = vkmodel.predict(x)
    ytrue   = vkmodel.predict(x)

    assert np.allclose(ytrue, ypred)
    
