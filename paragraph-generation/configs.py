import ml_collections
from datetime import datetime
import json


def get_config(args):
    """
        Configuration elements for related work paragraph generations
    """

    config = ml_collections.ConfigDict()

    # Environmental variables path for keys
    config.env_file_path = args.env_file_path

    # Experiment id
    config.exp_id = args.exp_id

    # Input instances file
    config.input_file = args.input_file

    # Examples file
    config.example_file = args.example_file

    # Intent file
    config.intent_file = args.intent_file

    # Categorical intent file
    config.categorical_intent_file = args.categorical_intent_file

    # Model type
    config.model_type = args.model_type

    # Model path
    config.model_path = args.model_path

    # Output path
    config.output_path = args.output_path + config.exp_id + '-' + config.model_type + '-' + datetime.now().strftime('%d.%m.%Y-%H:%M:%S') + '/'

    config.num_examples = args.num_examples

    config.citation_limit = args.citation_limit

    config.selection_mode = args.selection_mode

    config.prompt_plan = {'use_example': args.use_example,
                          'use_intent': args.use_intent,
                          'categorical_intent': args.categorical_intent}

    config.prompt_type = args.prompt_type

    if (config.prompt_plan['use_example']) and (config.prompt_plan['use_intent']):
        prompt_key = 'abstract-intent-example'
    elif (config.prompt_plan['use_example']) and (not config.prompt_plan['use_intent']):
        prompt_key = 'abstract-example'
    elif (not config.prompt_plan['use_example']) and (config.prompt_plan['use_intent']):
        prompt_key = 'abstract-intent'
    else:
        prompt_key = 'abstract'

    with open(args.prompt_file) as fr:
        prompts = json.load(fr)

    config.system_prompt = prompts[args.prompt_type][prompt_key]

    # The maximum number of tokens to generate
    config.max_new_tokens = args.max_new_tokens

    # Seed
    config.seed = args.seed

    # Batch size
    config.batch_size = args.batch_size

    return config
