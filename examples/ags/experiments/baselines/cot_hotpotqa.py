from examples.ags.scripts.operator import Operator
from examples.ags.scripts.graph import SolveGraph
from examples.ags.benchmark.hotpotqa import hotpotqa_evaluation
from examples.ags.scripts.operator_an import GenerateOp
from metagpt.actions.action_node import ActionNode 
from metagpt.configs.models_config import ModelsConfig
from metagpt.llm import LLM
from pydantic import BaseModel, Field
from typing import Tuple

HOTPOTQA_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Question: {question}
The revelant context: {context}
"""

class GenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")

class CoTGenerate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(name, llm)

    async def __call__(self, question: str, context: str, mode: str = None) -> Tuple[str, str]:
        thought = ""
        prompt = HOTPOTQA_PROMPT.format(question=question, context=context)
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        node = await ActionNode.from_pydantic(GenerateOp).fill(**fill_kwargs)
        response = node.instruct_content.model_dump()

        return response["answer"]

class CoTSolveGraph(SolveGraph):
    def __init__(self, name: str, llm_config, dataset: str):
        super().__init__(name, llm_config, dataset)
        self.cot_generate = CoTGenerate(self.llm)

    async def __call__(self, question: str, context: str) -> Tuple[str, str]:
        answer = await self.cot_generate(question, context, mode="context_fill")
        return answer, self.llm.cost_manager.total_cost

if __name__ == "__main__":
    async def main():
        llm_config = ModelsConfig.default().get("deepseek-chat")
        # llm_config = ModelsConfig.default().get("gpt-35-turbo-1106")
        graph = CoTSolveGraph(name="CoT", llm_config=llm_config, dataset="HotpotQA")
        file_path = "examples/ags/data/hotpotqa.jsonl"
        samples = 250 # 250 for validation, 1000 for test
        path = "examples/ags/data/baselines/general/hotpotqa"
        score = await hotpotqa_evaluation(graph, file_path, samples, path, test=False)
        return score

    import asyncio 
    asyncio.run(main())