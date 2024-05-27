import ml_collections
from datetime import datetime


def get_config(args):

    """
        Configuration elements for intention generations
    """

    config = ml_collections.ConfigDict()

    # Experiment id
    config.exp_id: str = args.exp_id

    """
        Model that will be used in generations. To replicate results: 
            google/flan-t5-xxl
    """
    config.model: str = args.model

    config.num_examples: int = args.num_examples

    config.custom_examples: bool = args.custom_examples

    # Input instances file
    config.input_file: str = args.input_file

    # Examples file
    config.example_file: str = args.example_file

    # Output path
    config.output_path: str = args.output_path + config.exp_id + '-' + config.model.split("/")[-1] + '-' + datetime.now().strftime("%d.%m.%Y-%H:%M:%S") +'/'

    """
        Generation strategies:
        Only one of the following can be chosen, otherwise exception will raise.
        
            greedy: Selects most probable token; sampling, top-k, num_beam, temperature,  will be deactivated.
            sampling: Sampling from top-k probable words without beam search.
            beam: Beam search without sampling.
            beam_sample: Beam search with sampling. You can check beam_sample() from huggingface
            contrastive: You can check contrastive_search() from huggingface
    """
    config.strategy: str = args.strategy

    # Prompts
    config.task_prefix: str = 'What is intention of the following paragraph?\n'
    config.task_suffix: str = ''

    # Temperature value for changing model creativity
    # Lies between 0 and 1, smaller values lead to more deterministic outputs when sampling is used.
    config.temperature: float = args.temperature

    # Number of beams
    config.num_beams: int = args.num_beams

    # The number of most probable words that will be used in sampling
    config.top_k: int = args.top_k

    # The value balancing the model confidence and the degeneration penalty in contrastive search decoding.
    config.penalty_alpha: float = args.penalty_alpha

    # The number of output sequences for a single input
    config.num_return_sequences: int = args.num_return_sequences

    # The maximum number of input tokens (prompt length)
    config.max_input_len: int = args.max_input_len
    
    # The maximum number of tokens to generate
    config.max_new_tokens: int = args.max_new_tokens

    # Batch size
    config.batch_size: int = args.batch_size

    return config

