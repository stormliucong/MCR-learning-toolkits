# main program to run the full pipeline
from models import EnhancedModel
from trainer import EnhancedModelTrainer
from generator import EnhancedModelGenerator
from utils.config import process_config, get_config_from_json
from utils.dirs import create_dirs
from utils.args import get_args
import os

def main():
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    # create the experiments dirs
    create_dirs([config.callbacks.checkpoint_dir])
    # set the number of GPU that will be used
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    print('create model!')
    model = EnhancedModel(config)
    print('create trainer!')
    trainer = EnhancedModelTrainer(model.model,config)
    print('Start training!')
    trainer.train()
    print('Start generating!')
    generator = EnhancedModelGenerator(config)

if __name__ == '__main__':
    main()
