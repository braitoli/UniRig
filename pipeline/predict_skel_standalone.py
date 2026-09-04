import os
import sys
import torch
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
import argparse
from pathlib import Path
from box import Box
import lightning as L

# Set up project root in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run import load
from src.data.extract import get_files
from src.data.datapath import Datapath
from src.tokenizer.spec import TokenizerConfig
from src.tokenizer.parse import get_tokenizer
from src.data.dataset import UniRigDatasetModule, DatasetConfig
from src.data.transform import TransformConfig
from src.model.parse import get_model
from src.system.parse import get_writer, get_system
from src.inference.download import download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--npz_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    task = load('task', args.task)

    files = get_files(
        data_name=task.components.data_name,
        inputs=args.input,
        input_dataset_dir=None,
        output_dataset_dir=args.npz_dir,
        force_override=True,
        warning=False,
    )
    datapath = Datapath(files=[f[1] for f in files], cls=None)

    data_config = load('data', os.path.join(str(ROOT_DIR), 'configs/data', task.components.data))
    transform_config = load('transform', os.path.join(str(ROOT_DIR), 'configs/transform', task.components.transform))
    tokenizer_config = load('tokenizer', os.path.join(str(ROOT_DIR), 'configs/tokenizer', task.components.tokenizer))

    parsed_tok_cfg = TokenizerConfig.parse(config=tokenizer_config)
    tokenizer = get_tokenizer(config=parsed_tok_cfg)

    predict_dataset_config = DatasetConfig.parse(config=data_config.get('predict_dataset_config')).split_by_cls()
    predict_transform_config = TransformConfig.parse(config=transform_config.get('predict_transform_config'))

    model_config = load('model', os.path.join(str(ROOT_DIR), 'configs/model', task.components.model))
    model = get_model(tokenizer=tokenizer, **model_config)

    data = UniRigDatasetModule(
        process_fn=model._process_fn,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        tokenizer_config=parsed_tok_cfg,
        debug=False,
        data_name=task.components.data_name,
        datapath=datapath,
        cls=None,
    )

    writer_config = dict(task.get('writer', {}))
    writer_config['npz_dir'] = args.npz_dir
    writer_config['output_dir'] = args.output_dir
    writer_config['user_mode'] = True
    callbacks = [get_writer(**writer_config, order_config=predict_transform_config.order_config)]

    resume_from_checkpoint = download(task.get('resume_from_checkpoint', None))

    system_config = load('system', os.path.join(str(ROOT_DIR), 'configs/system', task.components.system))
    system = get_system(
        **system_config,
        model=model,
        steps_per_epoch=1,
        optimizer_config=None,
        loss_config=None,
        scheduler_config=None
    )

    trainer_config = dict(task.get('trainer', {}))
    trainer = L.Trainer(callbacks=callbacks, logger=None, **trainer_config)

    print(f"[UniRig Skeleton] Running predict on {args.input}...")
    trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)
    print("[UniRig Skeleton] Prediction complete!")

if __name__ == "__main__":
    main()
