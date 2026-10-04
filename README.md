# Optimization Agent with Microsoft AutoGen

An LLM agent that uses tools to solve arithmetic and linear programming
problems (a SciPy solver and a calculator).

## Setup
1. python -m venv .venv
2. .\.venv\Scripts\Activate.ps1
3. pip install -r requirements.txt
4. Create a `.env` file with: GEMINI_API_KEY=your_key

## Run
python workflows\main.py

## Example questions
- What is the cost of 100 bags at 2500 rupees each with 1.5 rupees tax per bag?
- A factory makes chairs (profit $20) and tables (profit $30)... how many of each maximize profit?

## Limitations
- The solver handles maximization with <= constraints only
- No conversation memory (Gemini tool-call workaround)