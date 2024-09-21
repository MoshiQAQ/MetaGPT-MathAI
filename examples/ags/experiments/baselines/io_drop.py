from examples.ags.scripts.operator import Operator
from examples.ags.scripts.graph import SolveGraph
from examples.ags.benchmark.drop import drop_evaluation
from examples.ags.scripts.operator_an import GenerateOp
from metagpt.actions.action_node import ActionNode 
from metagpt.configs.models_config import ModelsConfig
from metagpt.llm import LLM
from pydantic import BaseModel, Field
from typing import Tuple

DROP_PROMPT = """
Your task: {inputs}
Write the final answer in the field of "answer". Keep the answer very concise and to the point.
"""


class GenerateOp(BaseModel):
    answer: str = Field(default="", description="The final answer to the question")

class IOGenerate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(name, llm)

    async def __call__(self, inputs: str, mode: str = None) -> Tuple[str, str]:

        prompt = DROP_PROMPT.format(inputs=inputs)
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        node = await ActionNode.from_pydantic(GenerateOp).fill(**fill_kwargs)
        response = node.instruct_content.model_dump()

        return response["answer"]

class IOSolveGraph(SolveGraph):
    def __init__(self, name: str, llm_config, dataset: str):
        super().__init__(name, llm_config, dataset)
        self.cot_generate = IOGenerate(self.llm)

    async def __call__(self, inputs: str) -> Tuple[str, str]:
        answer = await self.cot_generate(inputs, mode="context_fill")
        return answer, self.llm.cost_manager.total_cost

if __name__ == "__main__":
    async def main():
        llm_config = ModelsConfig.default().get("gpt-4o-mini")
        # llm_config = ModelsConfig.default().get("deepseek-chat")
        # llm_config = ModelsConfig.default().get("gpt-35-turbo")
        # llm_config = ModelsConfig.default().get("gpt-4o")
        graph = IOSolveGraph(name="IO", llm_config=llm_config, dataset="DROP")
        file_path = "examples/ags/data/drop_v0_dev.jsonl.gz"
        samples = 200
        path = "examples/ags/data/baselines/general/drop"
        score = await drop_evaluation(graph, file_path, samples, path, test=True)
        return score

    import asyncio 
    asyncio.run(main())