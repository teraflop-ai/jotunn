from typing import Optional, List

import daft
import pandas as pd
import timm
import torch
from daft import DataType
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from jotunn.components.distributed_base import Distributed


class PadToSize:
    def __init__(self, size, fill=(255, 255, 255), interpolation=Image.BILINEAR):
        self.size = size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        target_w, target_h = self.size[1], self.size[0]
        pad_w = max(target_w - w, 0)
        pad_h = max(target_h - h, 0)
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        return transforms.functional.pad(
            img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill
        )


def create_weeb_udf(
    model_name: str,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.python(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class WeebUDF:
        def __init__(self, model_name: str = model_name, device: str = "cuda"):
            self.device = torch.device(device=device)
            if self.device.type == "cpu":
                self.dtype = torch.float32
            else:
                self.dtype = torch.float16
            self.model = timm.create_model(f"hf-hub:{model_name}", pretrained=True).to(
                self.device, dtype=self.dtype
            )
            self.model.eval()

            self.transform = transforms.Compose(
                [
                    PadToSize(
                        size=(512, 512),
                        fill=(255, 255, 255),
                        interpolation=Image.BILINEAR,
                    ),
                    transforms.Resize(
                        (448, 448),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.CenterCrop((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[
                            0.48145467042922974,
                            0.45782750844955444,
                            0.40821072459220886,
                        ],
                        std=[
                            0.2686295509338379,
                            0.2613025903701782,
                            0.27577710151672363,
                        ],
                    ),
                ]
            )

            self.df_tags = pd.read_csv(
                hf_hub_download(
                    repo_id=model_name, repo_type="model", filename="selected_tags.csv"
                ),
                keep_default_na=False,
            )

        def __call__(self, images):
            inputs = torch.stack(
                [self.transform(Image.fromarray(img)) for img in images]
            ).to(self.device, dtype=self.dtype)

            with torch.no_grad():
                outputs = self.model(inputs)
                predictions = torch.sigmoid(outputs).cpu().numpy()

            results = []
            for i in range(predictions.shape[0]):
                pred = predictions[i]
                mask = pred >= self.df_tags["best_threshold"]
                tags = self.df_tags["name"][mask].tolist()
                scores = pred[mask].tolist()
                results.append({tag: float(score) for tag, score in zip(tags, scores)})
            return results

    return WeebUDF.with_init_args(model_name=model_name)


def create_florence_udf(
    model_name: str,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.python(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class FlorenceUDF:
        def __init__(
            self,
            model_name: str = model_name,
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
            task="<CAPTION>",
            device: str = "cuda",
        ):
            self.device = torch.device(device=device)
            if self.device.type == "cpu":
                self.dtype = torch.float32
            else:
                self.dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            self.processor = AutoProcessor.from_pretrained(model_name)

            self.model = Florence2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
            ).to(self.device)
            self.model = torch.compile(self.model)
            self.model.eval()

            self.max_new_tokens = max_new_tokens
            self.early_stopping = early_stopping
            self.do_sample = do_sample
            self.num_beams = num_beams
            self.task = task

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            inputs = (
                self.processor(
                    text=[self.task] * len(images),
                    images=images.to_pylist(),
                    return_tensors="pt",
                )
                .to(self.device)
                .to(self.dtype)
            )

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                early_stopping=self.early_stopping,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
            )

            texts = self.processor.batch_decode(
                outputs,
                skip_special_tokens=False,
            )

            formatted_texts = [
                self.processor.post_process_generation(text, task=self.task)
                for text in texts
            ]
            return formatted_texts

    return FlorenceUDF.with_init_args(
        model_name=model_name,
    )


def create_vllm_image_tagger_udf(
    model_name: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tags: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
):
    @daft.udf(
        return_dtype=DataType.string(),
        concurrency=concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
    )
    class VllmImageCaptionUDF:
        def __init__(
            self,
            model_name: str = model_name,
            prompt: str = prompt,
            max_tokens: int = max_tokens,
            temperature: float = temperature,
        ):
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            if tags:
                structured_outputs_params = StructuredOutputsParams(
                    json={
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string", "enum": tags},
                            }
                        },
                        "required": ["tags"],
                    }
                )
            else:
                structured_outputs_params = None

            self.vllm_engine = LLM(model=model_name, trust_remote_code=True)
            self.sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                structured_outputs=structured_outputs_params,
            )
            self.prompt = prompt

        def __call__(self, images: daft.DataFrame) -> daft.DataFrame:
            images = [Image.fromarray(img) for img in images]
            vllm_inputs = [
                {
                    "prompt": self.prompt,
                    "multi_modal_data": {"image": image},
                }
                for image in images
            ]
            outputs = self.vllm_engine.generate(vllm_inputs, self.sampling_params)
            generated_text = []
            for output in outputs:
                generated_text.append(output.outputs[0].text)
            return generated_text

    return VllmImageCaptionUDF.with_init_args(
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


class ImageTagger(Distributed):
    def __init__(
        self,
        model_name: str,
        tagger: str = "vllm",
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = 256,
        temperature: Optional[float] = 0.2,
        tags: Optional[List[str]] = None,
        batch_size: int = 1,
        input_column: str = "image",
        output_column: str = "tags",
        concurrency: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
    ):
        super().__init__(
            input_columns=input_column,
            output_column=output_column,
            batch_size=batch_size,
            concurrency=concurrency,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )
        self.tagger = tagger
        self.model_name = model_name
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tags = tags

    def _udf(self):
        if self.tagger == "vllm":
            return create_vllm_image_tagger_udf(
                model_name=self.model_name,
                prompt=self.prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tags=self.tags,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        elif self.tagger == "florence":
            return create_florence_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        elif self.tagger == "weeb":
            return create_weeb_udf(
                model_name=self.model_name,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
            )
        else:
            raise NotImplementedError()
