# Narayan

Knowledge assistant repo with 2 backend tracks:

- `backend/` legacy local RAG backend
- `backend_azure/` Azure-native upgrade backend

Use `backend_azure/` for Azure OpenAI + Azure AI Search showcase work.

## Azure setup summary

- One Azure student subscription is enough
- Create Azure AI Foundry project
- Create Azure OpenAI chat + embedding deployments
- Create Azure AI Search resource
- Copy `backend_azure/.env.example` to `backend_azure/.env`
- Create index with `python backend_azure/scripts/create_index.py`

## Cost guard

- Set Azure budget alert
- Keep chat model cheap
- Keep search tier minimal
- Use `backend_azure` for demo, not burn testing loops
