# pddlego-df
Iteratively predict PDDL domain file with LLMs

Code Usage

Run interactive_CC.ipynb

- Functions contain solver setting, OpenAI GPT setting, VAL setting, and other helper functions
  - In GPT set up: you could input your own OPENAI_API_KEY
  - In VAL set up:
    - Currently, I am using the absolute path. The common_path to each domain and problem files can be modified (or removed).
    - Parser and Validate are added in the folder.
- Frameworks contain the main algorithms
  - You can skip Basic Prompts. This contains only basic environment settings
  - Baseline: gpt directly generates plans
  - PDDL on CoinCollector: this is the main part!
    - Textworld_express setting: https://github.com/cognitiveailab/TextWorldExpress
    - directly run two cells will get the output including each step, large loop, small loop, its prompts, actions, simulation output...
