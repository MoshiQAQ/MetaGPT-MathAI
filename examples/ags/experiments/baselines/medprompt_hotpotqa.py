from examples.ags.scripts.operator import Operator
from examples.ags.scripts.graph import SolveGraph
from examples.ags.benchmark.hotpotqa import hotpotqa_evaluation
from examples.ags.scripts.operator_an import GenerateOp
from metagpt.actions.action_node import ActionNode 
from metagpt.configs.models_config import ModelsConfig
from metagpt.llm import LLM
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple
from collections import Counter

import random

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

MD_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
And the relevant context: {context}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the solution that is more capable of solving the problem compared to other solutions, as this is crucial for problem-solving.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

class MdEnsembleOp(BaseModel):
    thought: str = Field(
        default="",
        description="Step-by-step analysis of the solutions to determine the best one.",
    )
    solution_letter: str = Field(default="", description="The letter of the chosen best solution (only one letter).")

class MdEnsemble(Operator):
    """
    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm, name: str = "MdEnsemble", vote_count: int = 5):
        super().__init__(name, llm)
        self.vote_count = vote_count

    @staticmethod
    def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, str]]:
        shuffled_solutions = solutions.copy()
        random.shuffle(shuffled_solutions)
        answer_mapping = {chr(65 + i): solutions.index(solution) for i, solution in enumerate(shuffled_solutions)}
        return shuffled_solutions, answer_mapping

    async def __call__(self, solutions: List[str], problem: str, context:str, mode: str = None):
        print(f"solution count: {len(solutions)}")
        all_responses = []

        for _ in range(self.vote_count):
            shuffled_solutions, answer_mapping = self.shuffle_answers(solutions)

            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

            prompt = MD_ENSEMBLE_PROMPT.format(solutions=solution_text, question=problem, context=context)
            fill_kwargs = {"context": prompt, "llm": self.llm}
            if mode:
                fill_kwargs["mode"] = mode
            node = await ActionNode.from_pydantic(MdEnsembleOp).fill(**fill_kwargs)
            response = node.instruct_content.model_dump()

            answer = response.get("solution_letter", "A")
            answer = answer.strip().upper()

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        final_answer = solutions[most_frequent_index]
        return {"solution": final_answer}  

class MedPromptGraph(SolveGraph):
    def __init__(self, name: str, llm_config, dataset: str, vote_count: int = 5):
        super().__init__(name, llm_config, dataset)
        self.cot_generate = CoTGenerate(self.llm)
        self.md_ensemble = MdEnsemble(llm=self.llm, vote_count=vote_count)

    async def __call__(self, problem, context):
        solutions = []
        for i in range(3):
            solution = await self.cot_generate(problem, context, mode="context_fill")
            solutions.append(solution)
        solution = await self.md_ensemble(solutions, problem, context, mode="context_fill")
        return solution["solution"], self.llm.cost_manager.total_cost

if __name__ == "__main__":
    async def main():
        # llm_config = ModelsConfig.default().get("deepseek-coder")
        llm_config = ModelsConfig.default().get("gpt-4o-mini")
        # llm_config = ModelsConfig.default().get("gpt-35-turbo-1106")
        graph = MedPromptGraph(name="MedPrompt", llm_config=llm_config, dataset="HotpotQA")
        file_path = "examples/ags/data/hotpotqa.jsonl"
        samples = 1
        path = "examples/ags/data/baselines/general/hotpotqa"
        score, cost = await hotpotqa_evaluation(graph, file_path, samples, path, test=False)
        return score, cost

    import asyncio
    asyncio.run(main())