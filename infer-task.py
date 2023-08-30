import argparse
import numpy as np
import infer_web as iw

from configs.config import Config

def main():
    arg_parse()
    pass

def arg_parse() -> tuple:
    defCpuCount = int(np.ceil(Config().n_cpu / 1.5))

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, help="current task name") # 对应exp_dir1参数
    parser.add_argument("--sampling_rate", type=int, default=48000, choices=[32000,41000,48000], help="target model sampling rate, default=48000") # 对应sr2参数
    parser.add_argument("--f0_guide", type=bool, default=True, help="choose model have f0 guide, default=true") # 对应f0_3
    parser.add_argument("--f0_method", type=str, default="rmvpe_gpu", choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], help="f0 extract method, default=rmvpe_gpu") # 对应f0method8
    parser.add_argument("--train_input", type=str, required=True, help="train set input folder") # 对应trainset_dir4
    parser.add_argument("--spk_id", type=int, default=0, help="speaker id, default=0") # 对应spk_id5
    parser.add_argument("--cpu_count", type=int, default=defCpuCount, help="data process cpu's count, default=" + defCpuCount) # 对应np7
    parser.add_argument("--save_epoch", type=int, default=10, help="save model each epoch steps, default=10") # 对应save_epoch10
    parser.add_argument("--total_epoch", type=int, default=20, help="train total epoch steps") # 对应total_epoch11
    parser.add_argument("--batch_size", type=int, default=12, help="batch size for each gpu") # 对应batch_size12
    

    parser.add_argument("--protect", type=float, default=0.33, help="protect")

    args = parser.parse_args()
    sys.argv = sys.argv[:1]

    return args





if __name__ == "__main__":
    main()