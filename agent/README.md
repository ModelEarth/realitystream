# RealityStream AI Assistant Agent

This is a small Node.js AI assistant for RealityStream.

## Purpose

The assistant helps users:

- Generate `parameters.yaml` examples
- Recommend suitable ML models
- Understand local and Colab run steps
- Explain model outputs and feature importance

## Setup

```bash
cd agent
npm install
cp .env.example .env
```

Add your OpenAI API key in `.env`.

## Run

```bash
npm start
```

The server runs at:

```text
http://localhost:3000
```

## Test

```bash
curl -X POST http://localhost:3000/agent ^
-H "Content-Type: application/json" ^
-d "{\"message\":\"Create YAML for bee density prediction in Maine for 2021 using Random Forest\"}"
```

## Example Output

```yaml
folder: naics6-bees-counties-simple
features:
  data: industries
  state: ME
  startyear: 2021
  endyear: 2021
  naics:
    - 2
targets:
  data: bees
models: rfc
```