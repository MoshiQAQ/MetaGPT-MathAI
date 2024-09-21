from examples.ags.scripts.operator import Operator
from examples.ags.scripts.graph import SolveGraph
from examples.ags.benchmark.drop import drop_evaluation
from metagpt.actions.action_node import ActionNode 
from metagpt.configs.models_config import ModelsConfig
from metagpt.llm import LLM
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple

DROP_PROMPT = """
Your task: {inputs}
Think step by step, then write the final answer in the field of "answer".
"""

REVIEW_PROMPT = """
Given a knowledge-based question-answering task and a thoughtful solution, your task is to using critical thinking (questioning) to review the solution's correctness and provide a review result in boolean format.

your task: {inputs}
solution: {solution}

If you are more than 95 percent confident that the final answer is incorrect, please return False and give a feedback for the error. Otherwise, please return True and give a explanation for the correctness.
"""

REVISE_PROMPT = """
Given a knowledge-based question-answering task and a thoughtful solution which is just reviewed as incorrect, your task is to revise the solution to solve the question and ensure the final answer in solution field very concise.

your task: {inputs}
solution: {solution}
feedback: {feedback}
"""

class GenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")

class ReviewOp(BaseModel):
    feedback: str = Field(
        default="",
        description="Your FeedBack for this problem based on the criteria. If the review result is true, you can put it 'nothing here'.",
    )
    review_result: bool = Field(
        default=False,
        description="The Review Result (Bool). If you think this solution looks good for you, return 'true'; If not, return 'false'",
    )


class ReviseOp(BaseModel):
    solution: str = Field(default="", description="Based on the feedback, revised solution for this problem")


class CoTGenerate(Operator):
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
    
class Review(Operator):
    def __init__(self, llm: LLM, name: str = "Review"):
        super().__init__(name, llm)

    async def __call__(self, inputs, solution, mode: str = "context_fill"):
        prompt = REVIEW_PROMPT.format(inputs=inputs, solution=solution)
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        node = await ActionNode.from_pydantic(ReviewOp).fill(**fill_kwargs)
        response = node.instruct_content.model_dump()
        return response

class Revise(Operator):
    def __init__(self, llm: LLM, name: str = "Revise"):
        super().__init__(name, llm)

    async def __call__(self, inputs, solution, feedback, mode: str = "context_fill"):
        prompt = REVISE_PROMPT.format(inputs=inputs, solution=solution, feedback=feedback)
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        node = await ActionNode.from_pydantic(ReviseOp).fill(**fill_kwargs)
        response = node.instruct_content.model_dump()
        return response["solution"]

class SelfRefineGraph(SolveGraph):
    def __init__(self, name: str, llm_config, dataset: str):
        super().__init__(name, llm_config, dataset)
        self.cot_generate = CoTGenerate(self.llm)
        self.review = Review(self.llm)
        self.revise = Revise(self.llm)

    async def __call__(self, inputs):
        solution = await self.cot_generate(inputs, mode="context_fill")
        for i in range(3):
            review = await self.review(inputs, solution)
            if review["review_result"]:
                break
            solution = await self.revise(inputs, solution, review["feedback"])
        return solution, self.llm.cost_manager.total_cost

if __name__ == "__main__":
    async def main():
        llm_config = ModelsConfig.default().get("gpt-4o-mini")
        graph = SelfRefineGraph(name="self-refine", llm_config=llm_config, dataset="DROP")
        file_path = "examples/ags/data/drop_v0_dev.jsonl.gz"
        samples = 200
        path = "examples/ags/data/baselines/general"
        score = await drop_evaluation(graph, file_path, samples, path, test=True)
        return score

    import asyncio
    asyncio.run(main())
