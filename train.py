import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    args.eval = False
    args = postprocess_args(args)
    pprint(vars(args))

    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    print(args.save_path)



