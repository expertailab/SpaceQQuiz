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

## Question Generation Module
Create question-generation conda environment and install required libraries (for GPU use, check CUDA version):
```bash
conda create -n question-generation python=3.7
conda activate question-generation
cd SpaceQQuiz/question-generation
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

To run the question-generation module (You need to download the question-generation-squad-bart-large and question-generation-squad-t5-large):
```bash
python src/app.py
```

By default the endpoints will be:
* http://localhost:8080/generate_questions, question generation endpoint which receives a contexts and returns a question per each model (T5 and BART).

How to cite
-----------

To cite this research please use the following::
```
@inproceedings{garcia-silva-etal-2022-generating,
    title = "Generating Quizzes to Support Training on Quality Management and Assurance in Space Science and Engineering",
    author = "Garcia-Silva, Andres  and
      Berrio Aroca, Cristian  and
      Gomez-Perez, Jose Manuel  and
      Martinez, Jose  and
      Fleith, Patrick  and
      Scaglioni, Stefano",
    editor = "Shaikh, Samira  and
      Ferreira, Thiago  and
      Stent, Amanda",
    booktitle = "Proceedings of the 15th International Conference on Natural Language Generation: System Demonstrations",
    month = jul,
    year = "2022",
    address = "Waterville, Maine, USA and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.inlg-demos.2",
    pages = "4--6",
    abstract = "Quality management and assurance is key for space agencies to guarantee the success of space missions, which are high-risk and extremely costly. In this paper, we present a system to generate quizzes, a common resource to evaluate the effectiveness of training sessions, from documents about quality assurance procedures in the Space domain. Our system leverages state of the art auto-regressive models like T5 and BART to generate questions, and a RoBERTa model to extract answers for such questions, thus verifying their suitability.",
}
```
```
