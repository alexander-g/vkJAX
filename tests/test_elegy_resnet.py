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

    model   = elegy.Model(r18, run_eagerly=True)
    model.init(x)
    vkmodel = vkelegy.vkModel(r18)
    vkmodel.states = model.states
    vkmodel.initialized=True
    
    ypred   = vkmodel.predict(x)
    ytrue   = model.predict(x)

    assert np.allclose(ytrue, ypred, rtol=1e-4, atol=1e-5)
    
