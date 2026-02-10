from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
import modules.scripts as scripts
from modules import extra_networks
from modules.processing import StableDiffusionProcessing
import gradio as gr
from loractl.lib import utils, plot, lora_ctl_network, network_patch

class LoraCtlScript(scripts.Script):
    def __init__(self):
        self.original_network = None
        self.sampling_p = None
        super().__init__()

    sorting_priority = 10.2

    def title(self):
        return "Dynamic Lora Weights (reForge)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Dynamic Lora Weights", open=False):
                opt_enable = gr.Checkbox(
                    value=False, label="Enable Dynamic Lora Weights")
                opt_plot_lora_weight = gr.Checkbox(
                    value=False, label="Plot the LoRA weight in all steps")
        return [opt_enable, opt_plot_lora_weight]

    def process(self, p: StableDiffusionProcessing, opt_enable=True, opt_plot_lora_weight=False, **kwargs):
        self.original_network = extra_networks.extra_network_registry["lora"]
        if opt_enable and type(extra_networks.extra_network_registry["lora"]) != lora_ctl_network.LoraCtlNetwork:
            network = lora_ctl_network.LoraCtlNetwork()
            extra_networks.register_extra_network(network)
            extra_networks.register_extra_network_alias(network, "loractl")
        elif not opt_enable and type(extra_networks.extra_network_registry["lora"]) != lora_ctl_network.LoraCtlNetwork.__bases__[0]:
            extra_networks.register_extra_network(self.original_network)
            self.original_network = None
        network_patch.apply()
        utils.set_hires(False)
        utils.set_active(opt_enable)
        lora_ctl_network.reset_weights()
        plot.reset_plot()

        # Taken from https://github.com/hako-mikan/sd-webui-lora-block-weight/blob/b403cb7300f288360023c2a1978b86a361934e24/scripts/lora_block_weight.py#L436
        # This allows reloading lora weights at each step without modifying Classic/Neo's scripts and samplers code
        if not hasattr(self,"lbt_dr_callbacks"):
            self.lbt_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

    def denoiser_callback(self, params: CFGDenoiserParams):
        lora_network = extra_networks.extra_network_registry["lora"]
        if isinstance(lora_network, lora_ctl_network.LoraCtlNetwork):
            lora_network.reload_weights_for_step(params.denoiser.p, { "i": params.sampling_step, "x": params.x })

    def before_hr(self, p, *args):
        utils.set_hires(True)

    def postprocess(self, p, processed, opt_enable=True, opt_plot_lora_weight=False, **kwargs):
        if opt_plot_lora_weight and opt_enable:
            processed.images.extend([plot.make_plot()])
