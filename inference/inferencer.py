from copy import deepcopy
from typing import Dict, Any, List, Optional, Union

import torch
from PIL import Image

from .utils import denormalize_vae_output


def pil_img2rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB mode."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


class NaiveCache:
    """Key-value cache for inference.

    Attributes:
        key_cache: Dictionary storing key tensors per layer
        value_cache: Dictionary storing value tensors per layer
    """
    def __init__(self, num_layers: int):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    @property
    def seq_lens(self) -> int:
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


class UniXInferencer:
    """Core inference class for interleaved text-image generation.

    Attributes:
        model: Main model
        vae_model: VAE model for image generation
        tokenizer: Text tokenizer
        vae_transform: Transform for VAE processing
        vit_transform: Transform for ViT processing
        new_token_ids: Special token IDs
    """

    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def init_gen_context(self) -> Dict:
        """Initialize generation context.

        Returns:
            Dictionary with kv_lens, ropes, and past_key_values
        """
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text: str, gen_context: Dict) -> Dict:
        """Update context with text input.

        Args:
            text: Input text
            gen_context: Current generation context

        Returns:
            Updated generation context
        """
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def update_context_image(self, image: torch.Tensor, gen_context: Dict, vae: bool = True, vit: bool = True) -> Dict:
        """Update context with image input.

        Args:
            image: Input image tensor
            gen_context: Current generation context
            vae: Whether to process with VAE
            vit: Whether to process with ViT

        Returns:
            Updated generation context
        """
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)

        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def gen_image(
        self,
        image_shape: tuple,
        gen_context: Dict,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_text_precontext: Dict = None,
        cfg_img_precontext: Dict = None,
        cfg_interval: tuple = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
    ) -> Image.Image:
        """Generate image from context.

        Args:
            image_shape: Output image shape (H, W)
            gen_context: Current generation context
            cfg_text_scale: Classifier-free guidance scale for text
            cfg_img_scale: Classifier-free guidance scale for image
            cfg_text_precontext: Text precontext for CFG
            cfg_img_precontext: Image precontext for CFG
            cfg_interval: CFG interval
            cfg_renorm_min: Minimum renormalization value
            cfg_renorm_type: Renormalization type
            num_timesteps: Number of diffusion timesteps
            timestep_shift: Time shift for diffusion scheduler

        Returns:
            Generated PIL image
        """
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )

        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    @torch.no_grad()
    def decode_image(self, latent: torch.Tensor, image_shape: tuple) -> Image.Image:
        """Decode latent representation to image.

        Args:
            latent: VAE latent tensor
            image_shape: Target image shape (H, W)

        Returns:
            Decoded PIL image
        """
        h, w = image_shape
        h = h // self.model.latent_downsample
        w = w // self.model.latent_downsample
        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)

        if latent.dtype != self.vae_model.decoder.conv_in.weight.dtype:
            latent = latent.to(dtype=self.vae_model.decoder.conv_in.weight.dtype)

        image = self.vae_model.decode(latent)
        image = denormalize_vae_output(image, use_gen_normalization=True)

        # Convert to PIL Image
        image = image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        image = (image * 255).clip(0, 255).astype('uint8')
        image = Image.fromarray(image)

        return image

    @torch.no_grad()
    def gen_text(
        self,
        gen_context: Dict,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> str:
        """Generate text from context.

        Args:
            gen_context: Current generation context
            max_length: Maximum output length
            do_sample: Whether to use sampling
            temperature: Sampling temperature

        Returns:
            Generated text string
        """
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        if '<｜begin▁of▁sentence｜>' in output:
            parts = output.split('<｜begin▁of▁sentence｜>')
            if len(parts) >= 2:
                output = parts[-1]

        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        understanding_output: bool = False,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: list = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shapes: tuple = (512, 512),
        max_length: int = 500,
    ) -> List[Union[str, Image.Image]]:
        """Perform interleaved text-image inference.

        Args:
            input_lists: List of input text or images
            understanding_output: Whether to output understanding result
            do_sample: Whether to use sampling for text
            text_temperature: Text generation temperature
            cfg_text_scale: Text CFG scale
            cfg_img_scale: Image CFG scale
            cfg_interval: CFG interval
            timestep_shift: Diffusion timestep shift
            num_timesteps: Number of diffusion steps
            cfg_renorm_min: Minimum renormalization
            cfg_renorm_type: Renormalization type
            image_shapes: Default image shapes
            max_length: Maximum text generation length

        Returns:
            List of generated text or images
        """
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    if hasattr(self, 'vit_transform') and self.vit_transform is not None:
                        input_term = self.vit_transform(pil_img2rgb(input_term))

                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)
                    image_shapes = (input_term.shape[1], input_term.shape[2])

                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_length)
                output_list.append(gen_text)

            else:
                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)

        return output_list

    def __call__(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        **kargs,
    ) -> Dict[str, Any]:
        """Call interface for inference.

        Args:
            image: Input image
            text: Input text
            **kargs: Additional arguments

        Returns:
            Dictionary with 'image' and 'text' keys
        """
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
