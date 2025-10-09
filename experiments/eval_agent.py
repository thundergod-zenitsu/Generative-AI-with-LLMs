import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# --- Load Excel workbook ---
xls = pd.ExcelFile("template_agreement.xlsx")
section_dfs = {sheet: pd.read_excel(xls, sheet) for sheet in xls.sheet_names}

# --- Main LLM ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Create one DataFrame sub-agent per sheet ---
tools = []
for section, df in section_dfs.items():
    sub_agent = create_pandas_dataframe_agent(llm, df, verbose=False)
    tools.append(
        Tool.from_function(
            func=lambda query, sa=sub_agent: sa.invoke(query)["output"],
            name=f"{section}_Section_Agent",
            description=f"Use this tool to query the template sub-clauses in '{section}' section "
                        f"of the agreement. It includes columns Title, Content, Guidelines, Negotiation."
        )
    )

# --- Combine all section agents into one master agent ---
excel_agent = initialize_agent(
    tools, 
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


query = (
    "Find in the 'Termination' section the clause related to 'Termination for Convenience' "
    "and return its Title, Content, Guidelines, and Negotiation columns."
)
result = excel_agent.invoke(query)
print(result["output"])
