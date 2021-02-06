import elegy
import vkjax



class vkModel(elegy.Model):
    def _jit_functions(self):
        super()._jit_functions()

        self.call_pred_step_jit  = vkjax.wrap(self.call_pred_step_jit,  static_argnums=[1,3])
        self.call_test_step_jit  = vkjax.wrap(self.call_test_step_jit,  static_argnums=[2,6])
        self.call_train_step_jit = vkjax.wrap(self.call_train_step_jit, static_argnums=[2,6])




#def wrap_model(elegy_model: elegy.Model):
#    return vkModel(elegy_model)
