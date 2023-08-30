import argparse
import sys
import numpy as np
import infer_web as iw

from configs.config import Config


def main():
    args = arg_parse()
    iw.train1key(args.task_name, args.sampling_rate, args.f0_guide,
                 args.trainset_dir, args.spk_id, args.cpu_count,
                 args.f0_method, args.save_epoch, args.total_epoch,
                 args.batch_size, args.only_latest_ckpt, args.path_pretrained_g,
                 args.path_pretrained_d, args.gpus, args.cache_gpu,
                 args.save_ckpt_weight, args.version, args.gpus_rmvpe)
    pass


def arg_parse() -> tuple:
    defCpuCount = int(np.ceil(Config().n_cpu / 1.5))

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True,
                        help="current task name")  # 对应exp_dir1参数
    parser.add_argument("--sampling_rate", type=int, default=48000, choices=[
                        32000, 41000, 48000], help="target model sampling rate, default=48000")  # 对应sr2参数
    parser.add_argument("--f0_guide", type=bool, default=True,
                        help="choose model have f0 guide, default=true")  # 对应f0_3
    parser.add_argument("--f0_method", type=str, default="rmvpe_gpu", choices=[
                        "pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], help="f0 extract method, default=rmvpe_gpu")  # 对应f0method8
    parser.add_argument("--trainset_dir", type=str, required=True,
                        help="train set input folder")  # 对应trainset_dir4
    parser.add_argument("--spk_id", type=int, default=0,
                        help="speaker id, default=0")  # 对应spk_id5
    parser.add_argument("--cpu_count", type=int, default=defCpuCount,
                        help="data process cpu's count, default=" + defCpuCount)  # 对应np7
    parser.add_argument("--save_epoch", type=int, default=10,
                        help="save model each epoch steps, default=10")  # 对应save_epoch10
    parser.add_argument("--total_epoch", type=int, default=20,
                        help="train total epoch steps")  # 对应total_epoch11
    parser.add_argument("--batch_size", type=int, default=12,
                        help="batch size for each gpu")  # 对应batch_size12
    parser.add_argument("--only_latest_ckpt", type=bool, default=True,
                        help="only save latest ckpt file")  # 对应if_save_latest13
    parser.add_argument("--path_pretrained_g", type=str, default="assets/pretrained_v2/f0G48k.pth",
                        help="pretrain model G file path")  # 对应pretrained_G14
    parser.add_argument("--path_pretrained_d", type=str, default="assets/pretrained_v2/f0D48k.pth",
                        help="pretrain model G file path")  # 对应pretrained_D15
    parser.add_argument("--gpus", type=str, default="0",
                        help="use gpu number, default=0")  # 对应gpus16
    parser.add_argument("--cache_gpu", type=bool, default=False,
                        help="cache data to gpu memory, default=false")  # 对应if_cache_gpu17
    parser.add_argument("--save_ckpt_weight", type=bool, default=False,
                        help="save check point final model to weights folder")  # 对应if_save_every_weights18
    parser.add_argument("--version", type=str, choices=[
                        "v1", "v2"], default="v2", help="choose rvc version, default=v2")  # 对应version19
    parser.add_argument("--gpus_rmvpe", type=str, default="0-0",
                        help="rmvpe use gpu and threads, default=0-0")

    args = parser.parse_args()
    sys.argv = sys.argv[:1]

    return args


if __name__ == "__main__":
    main()
