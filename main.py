# main program to run the full pipeline
from src.models import EnhancedModel
from src.trainer import EnhancedModelTrainer
from src.generator import EnhancedModelGenerator
from utils.data_preprocessing import *
from utils.config import process_config, get_config_from_json
from utils.dirs import create_dirs
from utils.args import get_args
from utils.dictionary import build_dictionary
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

    # preprocess required data
    # build dictionary
    unique_concepts = get_intersect_concepts(config.data.csvpair, config.data.glove_condition_emb, config.data.glove_drug_emb)
    build_dictionary(unique_concepts, config.data.save_dir)

    # build weight matrices
    build_n2v_matrix(config.data.n2v_emb, os.path.join(config.data.save_dir, "concept2id"), config.data.save_dir)
    build_glove_matrix(config.data.glove_condition_emb, config.data.glove_drug_emb, 
    os.path.join(config.data.save_dir, "concept2id"), config.data.save_dir)

    # generate training pairs
    generate_training_pairs(config.data.csvpair, os.path.join(config.data.save_dir, "concept2id"), 
    config.data.glove_condition_emb, config.data.glove_drug_emb, config.data.save_dir)

    # split training pairs into batch
    split_into_batch(config.data.training_pairs, 500000, config.data.training_batch)

    # set the number of GPU that will be used
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    print('Create the model')
    enhanced_model = EnhancedModel(config)
    print('Create the trainer')
    trainer = EnhancedModelTrainer(enhanced_model.model, config)
    print('Start training')
    trainer.train()
    print('Start generating enhanced representations')
    generator = EnhancedModelGenerator(config)
    generator.generate_enhanced_rep()
    print('Enhanced representations have been generated')


if __name__ == '__main__':
    main()
