import elegy
import vkjax



class vkModel(elegy.Model):
    def jit_step(self):
        self.call_summary_step_jit = vkjax.wrap(
            self.call_summary_step,
            static_argnums=[2, 3],
        )
        self.call_pred_step_jit = vkjax.wrap(
            self.call_pred_step,
            static_argnums=[2, 3],
        )
        self.call_test_step_jit = vkjax.wrap(
            self.call_test_step,
            static_argnums=[5, 6],
        )
        self.call_train_step_jit = vkjax.wrap(
            self.call_train_step,
            static_argnums=[5, 6],
        )
        self.call_init_step_jit = vkjax.wrap(
            self.call_init_step,
            static_argnums=[],
        )

        self.jitted_members |= {
            "call_summary_step_jit",
            "call_pred_step_jit",
            "call_test_step_jit",
            "call_train_step_jit",
            "call_init_step_jit",
        }




#def wrap_model(elegy_model: elegy.Model):
#    return vkModel(elegy_model)
