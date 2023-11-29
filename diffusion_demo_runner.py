from data.segmentation import SegmentationDataset
from torch.utils.data import DataLoader
from diffusion.train_util import TrainLoop
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from diffusion.resample import create_named_schedule_sampler

model, diffusion = create_model_and_diffusion(
    **model_and_diffusion_defaults()
)
schedule_sampler = create_named_schedule_sampler(
    "uniform", diffusion,  maxt=1000)


dataset = SegmentationDataset(True, 5)
datal = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

TrainLoop(
    model=model.to("cuda:0"),
    diffusion=diffusion,
    classifier=None,
    dataloader=datal,
    batch_size=1,
    microbatch=-1,
    lr=1e-4,
    ema_rate=0.9999,
    log_interval=100,
    save_interval=5000,
    resume_checkpoint='',
    use_fp16=False,
    fp16_scale_growth=1e-3,
    schedule_sampler=schedule_sampler,
    weight_decay=0.0,
    lr_anneal_steps=0,
).run_loop(1)
