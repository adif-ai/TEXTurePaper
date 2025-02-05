from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers.utils import load_image
import shutil

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.training.views_dataset import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy

# import sd webui modules
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../..", "stable-diffusion-webui")
)
import sd_webui_modules


class TEXTure:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = self.cfg.optim.seed

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        if self.cfg.log.log_images:
            self.train_renders_path = make_path(self.exp_path / "vis" / "train")
            self.eval_renders_path = make_path(self.exp_path / "vis" / "eval")
        self.final_renders_path = self.exp_path

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / "config.yaml").open("w"))

        self.view_dirs = ["front", "left", "back", "right", "overhead", "bottom"]
        self.mesh_model = self.init_mesh_model()

        sd_webui_modules.sd_model_load(self.cfg.guide.diffusion_name)

        if str(self.cfg.guide.reference_image_path) != str(None):
            self.reference_image = load_image(self.cfg.guide.reference_image_path)

            if self.cfg.guide.reference_image_repeat != 1:
                self.reference_image = Image.fromarray(
                    torch.tile(
                        torch.from_numpy(np.array(self.reference_image)),
                        (
                            self.cfg.guide.reference_image_repeat,
                            self.cfg.guide.reference_image_repeat,
                            1,
                        ),
                    ).numpy()
                )
        else:
            self.reference_image = None

        self.dataloaders = self.init_dataloaders()
        self.back_im = (
            torch.Tensor(
                np.array(Image.open(self.cfg.guide.background_img).convert("RGB"))
            )
            .to(self.device)
            .permute(2, 0, 1)
            / 255.0
        )

        logger.info(f"Successfully initialized {self.cfg.log.exp_name}")

    def init_mesh_model(self) -> nn.Module:
        cache_path = Path("cache") / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(
            self.cfg.guide,
            device=self.device,
            render_grid_size=self.cfg.render.train_grid_size,
            cache_path=cache_path,
            texture_resolution=self.cfg.guide.texture_resolution,
            augmentations=False,
        )

        model = model.to(self.device)
        logger.info(
            f"Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}"
        )
        logger.info(model)
        return model

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        init_train_dataloader = MultiviewDataset(
            self.cfg.render, device=self.device
        ).dataloader()

        val_loader = ViewsDataset(
            self.cfg.render, device=self.device, size=self.cfg.log.eval_size
        ).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(
            self.cfg.render, device=self.device, size=self.cfg.log.full_eval_size
        ).dataloader()
        dataloaders = {
            "train": init_train_dataloader,
            "val": val_loader,
            "val_large": val_large_loader,
        }
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        )
        logger.add(
            lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format
        )
        logger.add(self.exp_path / "log.txt", colorize=False, format=log_format)

    def paint(self):
        logger.info("Starting training ^_^")
        # Evaluate the initialization
        if self.cfg.log.log_images:
            self.evaluate(self.dataloaders["val"], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(
            total=len(self.dataloaders["train"]),
            initial=self.paint_step,
            bar_format="{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for data in self.dataloaders["train"]:
            self.paint_step += 1
            pbar.update(1)
            self.paint_viewpoint(data)
            if self.cfg.log.log_images:
                self.evaluate(self.dataloaders["val"], self.eval_renders_path)
            self.mesh_model.train()

        self.mesh_model.change_default_to_median()
        logger.info("Finished Painting ^_^")
        logger.info("Saving the last result...")
        self.full_eval()

        if self.cfg.guide.upscale:
            logger.info("Upscale UV map...")
            self.upscale()

        logger.info("\tDone!")

    def upscale(self):
        uv_map = load_image(os.path.join(self.exp_path, "mesh", "albedo.png"))

        upscaled_uv_map = sd_webui_modules.upscaler_wrapper(
            image=uv_map,
            resize=self.cfg.guide.upscale_resize,
            upscaler1=self.cfg.guide.upscaler1,
        )[0]

        # shutil.move(
        #     os.path.join(self.exp_path, "mesh", "albedo.png"),
        #     os.path.join(self.exp_path, "mesh", "albedo_original.png"),
        # )
        os.remove(os.path.join(self.exp_path, "mesh", "albedo.png"))

        upscaled_uv_map.save(os.path.join(self.exp_path, "mesh", "albedo.png"))

    def evaluate(
        self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False
    ):
        logger.info(
            f"Evaluating and saving model, painting iteration #{self.paint_step}..."
        )
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(
                    save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg"
                )
                Image.fromarray(
                    (cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(
                        np.uint8
                    )
                ).save(save_path / f"{self.paint_step:04d}_{i:04d}_normals_cache.jpg")
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"{i:04d}_depth.pt")

        if not save_as_video:
            # Texture map is the same, so just take the last result
            texture = tensor2numpy(textures[0])
            Image.fromarray(texture).save(
                save_path / f"step_{self.paint_step:05d}_texture.png"
            )

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(
                save_path / "result.mp4",
                video,
                fps=25,
                quality=8,
                macro_block_size=1,
            )

            dump_vid(all_preds, "rgb")
        logger.info("Done!")

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders["val_large"], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / "mesh")
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    def make_inpaint_condition(self, image, image_mask):
        assert (
            image.size()[2:] == image_mask.size()[2:]
        ), "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        return image

    def paint_viewpoint(self, data: Dict[str, Any]):
        logger.info(f"--- Painting step #{self.paint_step} ---")
        theta, phi, radius = data["theta"], data["phi"], data["radius"]
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f"Painting from theta: {theta}, phi: {phi}, radius: {radius}")

        # Set background image
        if self.cfg.guide.use_background_color:
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else:
            background = F.interpolate(
                self.back_im.unsqueeze(0),
                (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                mode="bilinear",
                align_corners=False,
            )

        # if self.paint_step > 1:
        #     self.mesh_model.change_default_to_median()

        # Render from viewpoint
        outputs = self.mesh_model.render(
            theta=theta, phi=phi, radius=radius, background=background
        )
        render_cache = outputs["render_cache"]
        rgb_render_raw = outputs[
            "image"
        ]  # Render where missing values have special color
        depth_render = outputs["depth"]
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(
            background=background,
            render_cache=render_cache,
            use_median=self.paint_step > 1,
        )
        rgb_render = outputs["image"]
        # Render meta texture map
        meta_output = self.mesh_model.render(
            background=torch.Tensor([0, 0, 0]).to(self.device),
            use_meta_texture=True,
            render_cache=render_cache,
        )

        z_normals = outputs["normals"][:, -1:, :, :].clamp(0, 1)
        z_normals_cache = meta_output["image"].clamp(0, 1)
        edited_mask = meta_output["image"].clamp(0, 1)[:, 1:2]

        self.log_train_image(rgb_render, "rendered_input")
        self.log_train_image(depth_render[0, 0], "depth", colormap=True)
        self.log_train_image(z_normals[0, 0], "z_normals", colormap=True)
        self.log_train_image(z_normals_cache[0, 0], "z_normals_cache", colormap=True)

        update_mask, generate_mask, refine_mask = self.calculate_trimap(
            rgb_render_raw=rgb_render_raw,
            depth_render=depth_render,
            z_normals=z_normals,
            z_normals_cache=z_normals_cache,
            edited_mask=edited_mask,
            mask=outputs["mask"],
        )

        self.log_train_image(rgb_render * refine_mask, name="refine_regions")

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs["mask"][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render)
        resized_rgb_render = F.interpolate(
            cropped_rgb_render,
            (self.cfg.guide.image_resolution, self.cfg.guide.image_resolution),
            mode="bilinear",
            align_corners=False,
        )

        cropped_depth_render = crop(depth_render)
        cropped_depth_render = torch.cat(3 * [cropped_depth_render], dim=1)
        resized_depth_render = F.interpolate(
            cropped_depth_render,
            (self.cfg.guide.image_resolution, self.cfg.guide.image_resolution),
            mode="bilinear",
            align_corners=False,
        )

        cropped_update_mask = crop(update_mask)
        cropped_update_mask = torch.cat(3 * [cropped_update_mask], dim=1)
        resized_update_render = F.interpolate(
            cropped_update_mask,
            (self.cfg.guide.image_resolution, self.cfg.guide.image_resolution),
            mode="bilinear",
            align_corners=False,
        )
        resized_update_render[resized_update_render > 0] = 1

        self.log_train_image(cropped_rgb_render, name="cropped_input")

        # text embeddings
        dirs = data["dir"]
        view = self.view_dirs[dirs]
        prompt = self.cfg.guide.text + ", " + self.cfg.guide.added_text.format(view)
        negative_prompt = (
            self.cfg.guide.negative_text + ", " + self.cfg.guide.added_negative_text
        )

        logger.info(f"text: {prompt}")
        logger.info(f"negative text: {negative_prompt}")

        # checkerboard
        if (
            self.paint_step > 1
            and self.cfg.guide.use_refine
            and self.cfg.guide.use_checkerboard
        ):
            if self.cfg.guide.use_refine and self.cfg.guide.use_checkerboard:
                checker_mask = self.generate_checkerboard(
                    crop(update_mask), crop(refine_mask), crop(generate_mask)
                )
                self.log_train_image(
                    F.interpolate(
                        cropped_rgb_render,
                        (
                            self.cfg.guide.image_resolution,
                            self.cfg.guide.image_resolution,
                        ),
                    )
                    * (1 - checker_mask),
                    "checkerboard_input",
                )
                resized_update_render = resized_update_render + torch.cat(
                    3
                    * [
                        F.interpolate(
                            checker_mask,
                            size=(
                                self.cfg.guide.image_resolution,
                                self.cfg.guide.image_resolution,
                            ),
                        )
                    ],
                    dim=1,
                )
                resized_update_render[resized_update_render > 1] = 1

        if self.reference_image is None:
            pil_output = self.text_only_paint(
                crop,
                outputs,
                resized_update_render,
                resized_rgb_render,
                resized_depth_render,
                prompt,
                negative_prompt,
            )
        else:
            pil_output = self.reference_image_paint(
                crop,
                outputs,
                resized_update_render,
                resized_rgb_render,
                resized_depth_render,
                prompt,
                negative_prompt,
            )

        if self.cfg.log.log_images:
            pil_output.save(
                self.train_renders_path / f"{self.paint_step:04d}_direct_output.jpg"
            )

        cropped_rgb_output = np.array(pil_output).astype(np.float32) / 255.0
        cropped_rgb_output = np.expand_dims(cropped_rgb_output, axis=0).transpose(
            0, 3, 1, 2
        )
        cropped_rgb_output = torch.from_numpy(cropped_rgb_output).to(self.device)

        cropped_rgb_output = F.interpolate(
            cropped_rgb_output,
            (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Extend rgb_output to full image size
        rgb_output = rgb_render.clone()
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output
        self.log_train_image(rgb_output, name="full_output")

        # Project back
        object_mask = outputs["mask"]
        fitted_pred_rgb, _ = self.project_back(
            render_cache=render_cache,
            background=background,
            rgb_output=rgb_output,
            object_mask=object_mask,
            update_mask=update_mask,
            z_normals=z_normals,
            z_normals_cache=z_normals_cache,
        )
        self.log_train_image(fitted_pred_rgb, name="fitted")

        return

    def text_only_paint(
        self,
        crop,
        outputs,
        resized_update_render,
        resized_rgb_render,
        resized_depth_render,
        prompt,
        negative_prompt,
    ):
        object_mask = crop(outputs["mask"])
        object_mask = F.interpolate(
            object_mask,
            (self.cfg.guide.image_resolution, self.cfg.guide.image_resolution),
            mode="bilinear",
            align_corners=False,
        )
        object_mask[object_mask > 0] = 1
        resized_update_render[object_mask.expand_as(resized_update_render) == 0] = 1

        self.log_train_image(
            resized_rgb_render * (1 - resized_update_render), name="masked_input"
        )
        self.log_train_image(resized_depth_render, name="input_depth")
        self.log_train_image(resized_rgb_render, name="input_image")

        # txt2img
        if torch.all(resized_update_render):
            controlnets = [
                sd_webui_modules.depth_controlnet(
                    control_image=self.tensor_to_pil(resized_depth_render),
                    is_depth_map=True,
                ),
            ]

            pil_output = sd_webui_modules.txt2img_wrapper(
                width=self.cfg.guide.image_resolution,
                height=self.cfg.guide.image_resolution,
                prompt=prompt,
                negative_prompt=negative_prompt,
                controlnets=controlnets,
                seed=self.seed,
                steps=self.cfg.optim.steps,
            )[0]
        else:
            # make output with inpainting
            controlnets = [
                sd_webui_modules.depth_controlnet(
                    control_image=self.tensor_to_pil(resized_depth_render),
                    is_depth_map=True,
                ),
                sd_webui_modules.inpaint_controlnet(),
            ]

            pil_output = sd_webui_modules.img2img_inpaint_wrapper(
                width=self.cfg.guide.image_resolution,
                height=self.cfg.guide.image_resolution,
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_img=self.tensor_to_pil(resized_rgb_render),
                mask=self.tensor_to_pil(resized_update_render),
                controlnets=controlnets,
                seed=self.seed,
                steps=self.cfg.optim.steps,
                inpainting_fill=1,
                mask_blur=0,
                denoising_strength=self.cfg.guide.denoising_strength,
            )[0]

        return pil_output

    def reference_image_paint(
        self,
        crop,
        outputs,
        resized_update_render,
        resized_rgb_render,
        resized_depth_render,
        prompt,
        negative_prompt,
    ):
        object_mask = crop(outputs["mask"])
        object_mask = F.interpolate(
            object_mask,
            (self.cfg.guide.image_resolution, self.cfg.guide.image_resolution),
            mode="bilinear",
            align_corners=False,
        )
        object_mask[object_mask > 0] = 1

        ref_image = (
            np.array(
                self.reference_image.resize(
                    (
                        self.cfg.guide.image_resolution,
                        self.cfg.guide.image_resolution,
                    )
                )
            ).astype(np.float32)
            / 255.0
        )
        ref_image = np.expand_dims(ref_image, axis=0).transpose(0, 3, 1, 2)
        ref_image = torch.from_numpy(ref_image).to(self.device)

        # boundary_mask = self.dilate(object_mask, 10)
        # boundary_mask[object_mask == 1] = 0
        # boundary_mask[boundary_mask > 0] = 1
        # ref_image_mask = boundary_mask.clone()
        # ref_image_mask[resized_update_render[:, :1, :, :] == 1] = 1

        ref_image_mask = torch.zeros_like(object_mask)
        ref_image_mask[object_mask == 0] = 1
        ref_image_mask[resized_update_render[:, :1, :, :] == 1] = 1

        resized_rgb_render = (
            resized_rgb_render * (1 - ref_image_mask) + ref_image * ref_image_mask
        )

        # resized_update_render[object_mask.expand_as(resized_update_render) == 0] = 1
        self.log_train_image(
            resized_rgb_render * (1 - resized_update_render), name="masked_input"
        )
        self.log_train_image(resized_rgb_render, name="input_image")
        self.log_train_image(resized_depth_render, name="input_depth")

        # make output with inpainting
        controlnets = [
            sd_webui_modules.depth_controlnet(
                control_image=self.tensor_to_pil(resized_depth_render),
                is_depth_map=True,
            ),
            sd_webui_modules.inpaint_controlnet(),
            sd_webui_modules.reference_controlnet(
                control_image=self.reference_image,
                style_fidelity=self.cfg.guide.style_fidelity,
            ),
        ]

        pil_output = sd_webui_modules.img2img_inpaint_wrapper(
            width=self.cfg.guide.image_resolution,
            height=self.cfg.guide.image_resolution,
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_img=self.tensor_to_pil(resized_rgb_render),
            mask=self.tensor_to_pil(resized_update_render),
            controlnets=controlnets,
            seed=self.seed,
            steps=self.cfg.optim.steps,
            inpainting_fill=1,
            mask_blur=0,
            denoising_strength=self.cfg.guide.denoising_strength,
        )[0]

        return pil_output

    def eval_render(self, data):
        theta = data["theta"]
        phi = data["phi"]
        radius = data["radius"]
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(
            theta=theta, phi=phi, radius=radius, dims=(dim, dim), background="white"
        )
        z_normals = outputs["normals"][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs["image"]  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        diff = (
            (
                rgb_render.detach()
                - torch.tensor(self.mesh_model.default_color)
                .view(1, 3, 1, 1)
                .to(self.device)
            )
            .abs()
            .sum(axis=1)
        )
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = (
            rgb_render * (1 - uncolored_mask)
            + utils.color_with_shade(
                [0.85, 0.85, 0.85], z_normals=z_normals, light_coef=0.3
            )
            * uncolored_mask
        )

        outputs_with_median = self.mesh_model.render(
            theta=theta,
            phi=phi,
            radius=radius,
            dims=(dim, dim),
            use_median=True,
            render_cache=outputs["render_cache"],
        )

        meta_output = self.mesh_model.render(
            theta=theta,
            phi=phi,
            radius=radius,
            background=torch.Tensor([0, 0, 0]).to(self.device),
            use_meta_texture=True,
            render_cache=outputs["render_cache"],
        )
        pred_z_normals = meta_output["image"][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = (
            outputs_with_median["texture_map"]
            .permute(0, 2, 3, 1)
            .contiguous()
            .clamp(0, 1)
            .detach()
        )
        depth_render = outputs["depth"].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals

    def calculate_trimap(
        self,
        rgb_render_raw: torch.Tensor,
        depth_render: torch.Tensor,
        z_normals: torch.Tensor,
        z_normals_cache: torch.Tensor,
        edited_mask: torch.Tensor,
        mask: torch.Tensor,
    ):
        diff = (
            (
                rgb_render_raw.detach()
                - torch.tensor(self.mesh_model.default_color)
                .view(1, 3, 1, 1)
                .to(self.device)
            )
            .abs()
            .sum(axis=1)
        )
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)
        generate_mask = exact_generate_mask

        if self.cfg.guide.use_dilation:
            # Extend generate mask
            generate_mask = (
                torch.from_numpy(
                    cv2.dilate(
                        exact_generate_mask[0, 0].detach().cpu().numpy(),
                        np.ones((10, 10), np.uint8),
                    )
                )
                .to(exact_generate_mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )

        object_mask = torch.ones_like(generate_mask)
        object_mask[depth_render == 0] = 0

        # Generate the refine mask based on the z normals, and the edited mask

        refine_mask = torch.zeros_like(generate_mask)
        if self.cfg.guide.use_refine:
            if self.cfg.guide.z_update_abs:
                refine_mask[z_normals > self.cfg.guide.z_update_thr] = 1
            else:
                refine_mask[z_normals > z_normals_cache[:, :1, :, :] + 0.1] = 1
                refine_mask[object_mask == 0] = 0
            if self.cfg.guide.use_dilation:
                refine_mask = (
                    torch.from_numpy(
                        cv2.dilate(
                            refine_mask[0, 0].detach().cpu().numpy(),
                            np.ones((10, 10), np.uint8),
                        )
                    )
                    .to(mask.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            refine_mask[generate_mask == 1] = 0
        update_mask = generate_mask.clone()

        if self.cfg.guide.use_refine:
            update_mask[refine_mask == 1] = 1

        update_mask[object_mask == 0] = 0

        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(
                color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals
            )
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = (
                trimap_vis * (1 - exact_generate_mask)
                + utils.color_with_shade(
                    [255 / 255.0, 22 / 255.0, 67 / 255.0],
                    z_normals=z_normals,
                    light_coef=0.7,
                )
                * exact_generate_mask
            )

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = (
                shaded_rgb_vis * (1 - exact_generate_mask)
                + utils.color_with_shade(
                    [0.85, 0.85, 0.85], z_normals=z_normals, light_coef=0.7
                )
                * exact_generate_mask
            )

            if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(
                    color=[91 / 255.0, 155 / 255.0, 213 / 255.0], z_normals=z_normals
                )
                only_old_mask_for_vis = (
                    torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0)
                    .float()
                    .detach()
                )
                trimap_vis = trimap_vis * 0 + 1.0 * (
                    trimap_vis * (1 - only_old_mask_for_vis)
                    + refinement_color_shaded * only_old_mask_for_vis
                )
            self.log_train_image(shaded_rgb_vis, "shaded_input")
            self.log_train_image(trimap_vis, "trimap")

        return update_mask, generate_mask, refine_mask

    def generate_checkerboard(self, update_mask, refine_mask, generate_mask):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(
            checkerboard,
            (self.cfg.guide.image_resolution, self.cfg.guide.image_resolution),
        )
        checker_mask = F.interpolate(
            update_mask,
            (self.cfg.guide.image_resolution, self.cfg.guide.image_resolution),
        )
        only_old_mask = F.interpolate(
            torch.bitwise_and(refine_mask == 1, generate_mask == 0).float(),
            (self.cfg.guide.image_resolution, self.cfg.guide.image_resolution),
        )
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1]
        return checker_mask

    def project_back(
        self,
        render_cache: Dict[str, Any],
        background: Any,
        rgb_output: torch.Tensor,
        object_mask: torch.Tensor,
        update_mask: torch.Tensor,
        z_normals: torch.Tensor,
        z_normals_cache: torch.Tensor,
    ):
        render_update_mask = object_mask.clone()
        self.log_train_image(rgb_output * render_update_mask, "project_back_input")

        # Update the normals
        z_normals_cache[:, 0, :, :] = torch.max(
            z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :]
        )

        optimizer = torch.optim.Adam(
            self.mesh_model.get_params(),
            lr=self.cfg.optim.lr,
            betas=(0.9, 0.99),
            eps=1e-15,
        )
        for _ in tqdm(range(200), desc="fitting mesh colors"):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(
                background=background, render_cache=render_cache
            )
            rgb_render = outputs["image"]

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[
                :, :, mask > 0
            ]
            masked_mask = mask[mask > 0]
            loss = (
                (masked_pred - masked_target.detach()).pow(2) * masked_mask
            ).mean() + (
                (masked_pred - masked_pred.detach()).pow(2) * (1 - masked_mask)
            ).mean()

            meta_outputs = self.mesh_model.render(
                background=torch.Tensor([0, 0, 0]).to(self.device),
                use_meta_texture=True,
                render_cache=render_cache,
            )
            current_z_normals = meta_outputs["image"]
            current_z_mask = meta_outputs["mask"].flatten()
            masked_current_z_normals = current_z_normals.reshape(
                1, current_z_normals.shape[1], -1
            )[:, :, current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(
                1, z_normals_cache.shape[1], -1
            )[:, :, current_z_mask == 1][:, :1]
            loss += (
                (masked_current_z_normals - masked_last_z_normals.detach())
                .pow(2)
                .mean()
            )
            loss.backward()
            optimizer.step()

        return rgb_render, current_z_normals

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = (
                    einops.rearrange(tensor, "(1) c h w -> h w c")
                    .detach()
                    .cpu()
                    .numpy()
                )
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f"{self.paint_step:04d}_{name}.jpg"
            )

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = (
                self.train_renders_path / f"{self.paint_step:04d}_diffusion_steps"
            )
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(step_folder / f"{k:02d}_diffusion_step.jpg")

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            if tensor.shape[1] == 1:
                tensor = torch.cat(3 * [tensor], dim=1)
            Image.fromarray(
                (
                    einops.rearrange(tensor, "(1) c h w -> h w c")
                    .detach()
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)
            ).save(path)

    def tensor_to_pil(self, tensor: torch.Tensor):
        return Image.fromarray(
            (
                einops.rearrange(tensor, "(1) c h w -> h w c").detach().cpu().numpy()
                * 255
            ).astype(np.uint8)
        )

    def dilate(self, tensor: torch.Tensor, i: int):
        device = tensor.device
        return (
            torch.from_numpy(
                cv2.dilate(
                    tensor[0, 0].detach().cpu().numpy(), np.ones((i, i), np.uint8)
                )
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

    def blur(self, input_mask: torch.Tensor, i: int):
        blur_mask = self.tensor_to_pil(input_mask).filter(ImageFilter.BoxBlur(i))
        blur_array = np.array(blur_mask).astype(np.float32) / 255.0
        blur_array = np.expand_dims(blur_array, axis=0).transpose(0, 3, 1, 2)
        blur_array = torch.from_numpy(blur_array).to(self.device)
        input_mask[blur_array < 0.15] = 0
        return input_mask
