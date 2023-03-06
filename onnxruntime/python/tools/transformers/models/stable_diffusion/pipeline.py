import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from diffusers.pipelines.stable_diffusion import OnnxStableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.utils import logging
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from onnxruntime.transformers.io_binding_helper import TypeHelper
from onnxruntime import OrtValue

logger = logging.get_logger(__name__)


class OrtModelBinding:
    def __init__(self, ort_session, io_shape, device_id=0):
        for input in ort_session.get_inputs():
            assert input.name in io_shape
        for output in ort_session.get_outputs():
            assert output.name in io_shape
        input_names = [input.name for input in ort_session.get_inputs()]
        output_names = [output.name for output in ort_session.get_outputs()]

        self.io_shape = io_shape
        self.io_numpy_type = TypeHelper.get_io_numpy_type_map(ort_session)
        self.io_binding = ort_session.io_binding()
        self.io_ort_value = {}

        for name in input_names + output_names:
            ort_value = OrtValue.ortvalue_from_shape_and_type(
                io_shape[name], self.io_numpy_type[name], "cuda", device_id
            )
            self.io_ort_value[name] = ort_value

            if name in output_names:
                self.io_binding.bind_ortvalue_output(name, ort_value)
            else:
                self.io_binding.bind_ortvalue_input(name, ort_value)


class OrtStableDiffusionPipeline(OnnxStableDiffusionPipeline):
    def __init__(
        self,
        vae_encoder: OnnxRuntimeModel,
        vae_decoder: OnnxRuntimeModel,
        text_encoder: OnnxRuntimeModel,
        tokenizer: CLIPTokenizer,
        unet: OnnxRuntimeModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: OnnxRuntimeModel,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae_encoder,
            vae_decoder,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        self.model_bindings = {}

    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

        if not np.array_equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]
        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_io_binding(self, batch, height, width, encoder_hidden_states, device_id=0):
        """Returnas IO binding object for a session. For CUDA only
        batch == batch_size * num_images_per_prompt
        """

        # Re-use existing buffers if buffer size does not change.
        if "unet" in self.model_bindings:
            if self.model_bindings["unet"].io_shape["encoder_hidden_states"] == list(encoder_hidden_states.shape):
                return

        io_shape = {
            "sample": [2 * batch, 4, height // 8, width // 8],
            "timestep": [1],
            "encoder_hidden_states": list(encoder_hidden_states.shape),  # (2B, S, H)
            "out_sample": [2 * batch, 4, height // 8, width // 8],
        }
        self.model_bindings["unet"] = OrtModelBinding(self.unet.model, io_shape, device_id)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # Extra constraints
        if guidance_scale <= 1.0:
            raise ValueError(f"Only support classifier free guidance (guidance_scale > 1.0).")

        if generator is None:
            generator = np.random

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        self.prepare_io_binding(batch_size * num_images_per_prompt, height, width, prompt_embeds, device_id=0)
        self.model_bindings["unet"].io_ort_value["encoder_hidden_states"].update_inplace(prompt_embeds)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            latent_model_input = np.concatenate([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.numpy()
            self.model_bindings["unet"].io_ort_value["sample"].update_inplace(latent_model_input)

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            self.model_bindings["unet"].io_ort_value["timestep"].update_inplace(timestep)

            self.unet.model.run_with_iobinding(self.model_bindings["unet"].io_binding)
            noise_pred = self.model_bindings["unet"].io_ort_value["out_sample"].numpy()

            # perform guidance
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        # image = self.vae_decoder(latent_sample=latents)[0]
        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate([self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])])

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="np"
            ).pixel_values.astype(image.dtype)

            images, has_nsfw_concept = [], []
            for i in range(image.shape[0]):
                image_i, has_nsfw_concept_i = self.safety_checker(
                    clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                )
                images.append(image_i)
                has_nsfw_concept.append(has_nsfw_concept_i[0])
            image = np.concatenate(images)
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
