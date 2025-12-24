I will restructure the CSO agent system to support the 6 requested agents using the provided prompts.

**1. Create/Update Agent Files**
I will create or update the following files in `src/application/agents/cso/`. Each agent will be implemented using `PydanticChatAgent` and `AgentConfig`, deriving the `role`, `goal`, and `backstory` from the provided prompts.

* **`strategy_agent.py`** (Update): Use `STRATEGY_MODE_PROMPT`.

* **`value_prop_agent.py`** (Create): Use `VALUE_PROP_MODE_PROMPT`.

* **`gtm_agent.py`** (Create): Use `GTM_MODE_PROMPT`.

* **`railroad_intel_agent.py`** (Create): Use `RAILROAD_INTEL_MODE_PROMPT`.

* **`mna_agent.py`** (Create): Use `MNA_MODE_PROMPT`.

* **`artifact_agent.py`** (Create): Use `ARTIFACT_MODE_PROMPT`.

**2. Update** **`router_agent.py`**

* Import the 6 agents.

* Update `CSORouterAgent` to initialize: `strategy`, `value_prop`, `gtm`, `railroad_intel`, `mna`, `artifact`.

* Update `agent_descriptions_map` with concise descriptions extracted from the prompts to ensure accurate routing.

* Remove `risk` and `execution` agents.

**3. Cleanup**

* Delete `risk_agent.py` and `execution_agent.py`.

**4. Verification**

* I will verify the `router_agent.py` imports and initializes the new agents correctly.

