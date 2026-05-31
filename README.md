## Parkinson's Classification using WearGaitPD Dataset

This project is setup with [uv](https://docs.astral.sh/uv/). Please visit the website and install it onto your system. Once installed, you can get all the dependencies by running `uv sync`.

To add dependencies, use `uv add`. For any other issues, please check the documentation, or ask your favourite LLM.

### Access token for Synapse

Go to the settings page on Synapse and create a personal access token (PAT) for yourself. This is used to download the data from their servers. Then, create a file called `.env` in the project root, and add the environment variable `PAT`.

```text
PAT="yourPersonalAccessCode"
```
