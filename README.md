---
title: NLPinitiative Streamlit App
emoji: 🐨
colorFrom: indigo
colorTo: yellow
sdk: streamlit
sdk_version: 1.42.2
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# Setup for Pushing to GitHub and HF Space:
#### Due to the project being configured to use hugging face spaces to host the python web-app, the instructions will outline how to setup the project to push to any newly created Hugging Face Space.
Note: Streamlit can still be developed and deployed to environments other than Hugging Face Spaces. Refer to the appropriate documentation associated with a chosen hosting service for how to deploy the web-app to the services environment.

### After Creation of a Streamlit Hugging Face Space:
 - In the directory of the cloned repository, add the hugging face space as an additional remote origin:
    - You can specify any name to use for the origin name (i.e., hf_origin)
```
git remote add <hf-origin-name> <hf-space-url>
```

 - Once the space is linked, you will need to force update the space with the contents of the current repository as follows (This will sync the HF Space with the main repositories history):
```
git push --force <hf-origin-name> main
```

 - Following these steps, any new commits made can be pushed to the HF Space by using the following command:
```
git push <hf-space-name> main
```

