# NLPinitiative Streamlit Documentation 

Codebase for the Streamlit app hosted on Hugging Face Spaces that provides a basic user interface for performing inference on text input by the user using the models training within the NLPinitiative project.

---

## Project Details

### Description

The NLPinitiative Discriminatory Text Classifier is an advanced natural language processing tool designed to detect and flag potentially discriminatory or harmful language. By analyzing text for biased, offensive, or exclusionary content, this classifier helps promote more inclusive and respectful communication. Simply enter your text below, and the model will assess it based on linguistic patterns and context. While the tool provides valuable insights, we encourage users to review flagged content thoughtfully and consider context when interpreting results.

This project was developed as part of a sponsored project for the **<a href="https://www.j-initiative.org/" style="text-decoration:none">The J-Healthcare Initiative</a>** for the purpose of detecting discriminatory speech from public officials and news agencies targetting marginalized communities communities.

---

### How The Tool Works

The application utilizes two fine-tuned NLP models: 

- A binary classifier for classifying input as Discriminatory or Non-Discriminatory (prediction classes of 1 and 0 respectively).
- A multilabel regression model for assessing the likelihood of specific categories of discrimination 
   (Gender, Race, Sexuality, Disability, Religion and Unspecified) from a value of 0.0 (no confidence) and 1.0 (max confidence).

Both models are use the pretrained **<a href="https://doi.org/10.48550/arXiv.1810.04805" style="text-decoration:none">BERT</a>** (Bidirectional Encoder Representations from Transformers) as the base model, which was trained using the master dataset (which can be viewed on the Datasets tab). The master dataset includes data extractedand reformatted for use in training these models from the **<a href="https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset" style="text-decoration:none">ETHOS dataset</a>** and the **<a href="https://github.com/marcoguerini/CONAN?tab=readme-ov-file#multitarget-conan" style="text-decoration:none">Multitarget-CONAN dataset</a>**.

---

### Project Links
* **<a href="https://github.com/dlsmallw/NLPinitiative" style="text-decoration:none"><img src="https://raw.githubusercontent.com/tandpfun/skill-icons/refs/heads/main/icons/Github-Dark.svg" style="margin-right: 3px;" width="20" height="20"/> NLPinitiative GitHub Project</a>**  - The training/evaluation pipeline used for fine-tuning the models.
* **<a href="https://huggingface.co/{BIN_REPO}" style="text-decoration:none">🤗 NLPinitiative HF Binary Classification Model Repository</a>** - The Hugging Face hosted Binary Classification Model Repository.
* **<a href="https://huggingface.co/{ML_REPO}" style="text-decoration:none">🤗 NLPinitiative HF Multilabel Regression Model Repository</a>** - The Hugging Face hosted Multilabel Regression Model Repository.
* **<a href="https://huggingface.co/{DATASET_REPO}" style="text-decoration:none">🤗 NLPinitiative HF Dataset Repository</a>** - The Hugging Face hosted Dataset Repository.

Codebase for the Streamlit app hosted on Hugging Face Spaces that provides a basic user interface for performing inference on text input by the user using the models training within the NLPinitiative project.

---

### Project Setup

For the purposes of building and running the project for development, a bash script `setup.sh` has been created to make the process of performing various development operations (defined below). Use of this script will require use of a bash shell (git bash for windows users).

This script can be activated by using `source ./setup.sh` while within the project source directory.

#### Commands

 - `build`: This will setup a virtual environment within the project source directory and install all necessary dependencies for development.
 - `clean`: This will deactivate the virutal environment, and remove the .venv directory (uninstalling all dependencies).
 - `requirements` - *Important when pushing the codebase to the HF Space*: This will generate/update the `requirements.txt` file containing the required dependencies for the project.
    - This is required for the HF space to properly download dependencies due to using `pip` for initializing the application.
 - `docs build`: Parses the docstrings in the project and generates the project documentation using mkdocs.
 - `docs serve`: Serves the mkdocs documentation to a local dev server that can be opened in a browser.
 - `docs deploy`: Deploys the mkdocs documentation to the linked GitHub repositories 'GitHub Pages'.
 - `run dev`: Runs the Streamlit application locally for monitored development.
 - `set bin_repo <HF Model Repository>`: Sets the binary model repository ID to the specified string.
    - This is the source for downloading the model tensor file.
 - `set ml_repo <HF Model Repository>`: Sets the multilabel regression model repository ID to the specified string.
    - This is the source for downloading the model tensor file.
 - `set ds_repo <HF Dataset Repository>`: Sets the dataset repository ID to the specified string.
    - This is the source for downloading the datasets.

#### HF Spaces Setup

Due to the project being configured to use hugging face spaces to host the python web-app, the instructions will outline how to setup the project to push to any newly created Hugging Face Space.

**Note**: Streamlit can still be developed and deployed to environments other than Hugging Face Spaces. Refer to the appropriate documentation associated with a chosen hosting service for how to deploy the web-app to the services environment.

**After Creation of a Streamlit Hugging Face Space**:

In the directory of the cloned repository, add the hugging face space as an additional remote origin: 
`git remote add <hf-origin-name> <hf-space-url>`

 - **NOTE**: *You can specify any name to use for the origin name (i.e., hf_origin)*

Once the space is linked, you will need to force update the space with the contents of the current repository as follows (This will sync the HF Space with the main repositories history):
`git push --force <hf-origin-name> main`

Following these steps, any new commits made can be pushed to the HF Space by using the following command:
`git push <hf-space-name> main`

---

### Project layout

```
├── docs                <- A directory containing documentation used for generating and serving 
│                          project documentation
├── scripts             <- Source code for model inference               
│      ├── __init__.py         <- Makes modeling a Python module    
│      ├── config.py           <- Store useful variables and configuration
│      └── predict.py          <- Code to run model inference with trained models
├── app.py              <- Entry point for the application
├── config.toml         <- Stores HF repository information
├── LICENSE             <- Open-source license if one is chosen
├── mkdocs.yml          <- mkdocs project configuration
├── Pipfile             <- The project dependency file for reproducing the analysis environment, 
│                          e.g., generated with `pipenv install`
├── Pipfile.lock        <- Locked file containing hashes for dependencies
├── README.md           <- The top-level README for developers using this project
├── requirements.txt    <- Plaintext dependency information (necessary for app hosting)
└── setup.sh            <- Bash script containing convenience commands for managing the project
```
