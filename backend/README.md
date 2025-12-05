# AI C-Suite Agent SaaS Backend

## Setup

1. Install dependencies:
   ```bash
   pip install poetry
   poetry install
   ```

2. Environment Variables:
   Copy `.env.example` to `.env` and fill in the values.

3. Run the server:
   ```bash
   poetry run uvicorn src.main:app --reload
   ```

## Architecture

This project follows a layered hexagonal architecture.
- **API Layer**: `src/api`
- **Application Layer**: `src/application`
- **Domain Layer**: `src/domain`
- **Infrastructure Layer**: `src/infrastructure`

See `task` file for detailed architecture documentation.
