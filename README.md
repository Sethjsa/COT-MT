# COT-MT

## Translations
Step-by-step and Multi-pass (Translate Again) translations and evaluations per step are available under Results.

## Data
We use WMT24++, found in the required format in Dataset.

## Running jobs
We use a separate script for running each type of reasoning method: for Step-by-step, use `async_call_step_by_step_gem-gpt.py`, and for Translate Again use `async_call_simple_updated.py`. As it stands, data and models are set manually inside the scripts.

For more information, please refer to our EMNLP Paper [here](https://arxiv.org/abs/2506.04521), or contact the authors at s.aycock [at] uva.nl
