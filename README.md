# pddlego-df
Iteratively predict PDDL domain file with LLMs

## Installation and Setup

### 1. Python Version
It is recommended to use Python 3.8+ (though 3.7+ will likely work).

### 2. Required Python Packages
Install the following packages via pip:

```bash
pip install requests
pip install pandas
pip install openai
pip install backoff
pip install textworld-express
```

- requests: For making HTTP requests (used by the OpenAI / other APIs).
- pandas: For data manipulation and analysis.
- openai: For interacting with the OpenAI API (GPT models, etc.).
- backoff: For handling retry logic (especially helpful with API rate limits).
- textworld-express: For the Coin Collector environment and text-based environment simulations.
  
### 3. VAL (Parser/Validate) Setup
This project uses VAL for parsing and validating PDDL domain/problem files.

a. Obtain VAL: Clone the VAL repository from GitHub and compile it:

```bash
git clone https://github.com/KCL-Planning/VAL.git
cd VAL
make
```

The build artifacts (e.g., Parser, Validate) will typically appear under ./build/<platform>/<build-type>/bin/.

b. Update Paths:

- By default, the code uses an absolute path for the Parser executable.
- In your code, set parser_path (and similarly for Validate, if you use it) to point to where VAL’s Parser and Validate binaries are located.

### 4. OpenAI API Key
If you plan to use OpenAI’s GPT models:

- Sign up for an OpenAI account.
- Retrieve your OpenAI API Key from the API Keys page.
- In your code (or environment variables), set openai.api_key = "YOUR_API_KEY" or use an environment variable like:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

## Code Usage
### 1. Run interactive_CC.ipynb
Run this notebook in compiler e.g. VSCode for close source models

- This notebook demonstrates usage of:
  - The solver settings
  - OpenAI GPT settings
  - VAL setup
  - Other helper functions
    
- Modifying the common_path:
  - You can edit common_path in the notebook or Python scripts to point to your domain/problem files or to reference them differently (relative path instead of absolute).
    
- PDDL on CoinCollector:
  - The main example is in the “PDDL on CoinCollector” section, which uses TextWorldExpress.
  - Simply run the relevant cells to see how the environment is created, how the steps and loops are executed, and how prompts and actions are handled.

- Frameworks in the Repository
  - Basic Prompts: Contains fundamental environment settings.
  - Baseline: Shows how GPT can directly generate plans.
  - PDDL on CoinCollector: The primary workflow that uses GPT to generate domain files and/or plans, then tests them in the TextWorldExpress environment.

### 2. Run interactive_CC_server.py
Run this python file on server for open source models


