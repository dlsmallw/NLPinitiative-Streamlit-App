# NLPinitiative Streamlit Documentation 

---

## Project Details

### Description

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
