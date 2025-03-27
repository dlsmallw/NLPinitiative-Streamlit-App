# NLPinitiative Streamlit Documentation 

---

## Project Details

### Description

Codebase for the Streamlit app hosted on Hugging Face Spaces that provides a basic user interface for performing inference on text input by the user using the models training within the NLPinitiative project.

---
### Project Setup

**Setup for Pushing to GitHub and HF Space**:

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
│      └── predict.py          <- Code to run model inference with trained models
├── app.py              <- Entry point for the application
├── config.py           <- Store useful variables and configuration
├── LICENSE             <- Open-source license if one is chosen
├── mkdocs.yml          <- mkdocs project configuration
├── Pipfile             <- The project dependency file for reproducing the analysis environment, 
│                          e.g., generated with `pipenv install`
├── Pipfile.lock        <- Locked file containing hashes for dependencies
├── README.md           <- The top-level README for developers using this project
├── requirements.txt    <- Plaintext dependency information (necessary for app hosting)
└── setup.sh            <- Bash script containing convenience commands for managing the project
```
