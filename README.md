# SpaceQQuiz

SpaceQQuiz is a system to generate quizzes, a common resource to evaluate training sessions, out of quality procedure documents in the Space domain. Our system leverages state of the art auto-regressive models like [T5](https://arxiv.org/pdf/1910.10683.pdf) and [BART](https://arxiv.org/abs/1910.13461) to generate questions, and a [RoBERTa](https://arxiv.org/abs/1907.11692) model to extract answer for the questions, thus verifying their suitability.

## Requirements:
* [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)

## Installation:
Create a new conda environment:
```bash
conda create -n spaceqquiz python=3.9
conda activate spaceqquiz
cd SpaceQQuiz/
pip install -r requirements.txt
```

## Execution
```bash
streamlit run run_question_generation.py -- --question_generation_endpoint=$QUESTION_GENERATION_ENDPOINT
```
